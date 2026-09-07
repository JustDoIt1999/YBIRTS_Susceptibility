from __future__ import annotations

import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from Susceptibility_Mapping.su_prediction_mapping import (
    ARCGIS_QTP_ALBERS_WKT,
    MappingError,
    ResultSpec,
    filter_output_to_prediction_domain,
    map_predictions,
    parse_result_spec,
)


SOURCE_QTP_ALBERS_WKT = (
    'PROJCS["QTP_Albers_Equal_Area",GEOGCS["WGS 84",'
    'DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,'
    'AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],'
    'PRIMEM["Greenwich",0],UNIT["Degree",0.0174532925199433]],'
    'PROJECTION["Albers_Conic_Equal_Area"],PARAMETER["latitude_of_center",0],'
    'PARAMETER["longitude_of_center",90],PARAMETER["standard_parallel_1",27.5],'
    'PARAMETER["standard_parallel_2",37.5],PARAMETER["false_easting",0],'
    'PARAMETER["false_northing",0],UNIT["metre",1,'
    'AUTHORITY["EPSG","9001"]],AXIS["Easting",EAST],AXIS["Northing",NORTH]]'
)


def build_small_gpkg(path: Path) -> None:
    with closing(sqlite3.connect(path)) as conn:
        conn.executescript(
            f"""
            CREATE TABLE gpkg_contents (
                table_name TEXT PRIMARY KEY,
                data_type TEXT NOT NULL,
                identifier TEXT,
                description TEXT DEFAULT '',
                last_change DATETIME NOT NULL,
                min_x DOUBLE,
                min_y DOUBLE,
                max_x DOUBLE,
                max_y DOUBLE,
                srs_id INTEGER
            );
            CREATE TABLE gpkg_geometry_columns (
                table_name TEXT PRIMARY KEY,
                column_name TEXT NOT NULL,
                geometry_type_name TEXT NOT NULL,
                srs_id INTEGER NOT NULL,
                z TINYINT NOT NULL,
                m TINYINT NOT NULL
            );
            CREATE TABLE gpkg_spatial_ref_sys (
                srs_name TEXT NOT NULL,
                srs_id INTEGER NOT NULL PRIMARY KEY,
                organization TEXT NOT NULL,
                organization_coordsys_id INTEGER NOT NULL,
                definition TEXT NOT NULL,
                description TEXT
            );
            CREATE TABLE qtp_slope_units (
                fid INTEGER PRIMARY KEY,
                geom MULTIPOLYGON,
                basin_id TEXT,
                su_id TEXT
            );
            CREATE TRIGGER rtree_like_update_trigger
            AFTER UPDATE ON qtp_slope_units
            WHEN OLD.fid != NEW.fid AND (NEW.geom IS NULL OR ST_IsEmpty(NEW.geom))
            BEGIN SELECT 1; END;
            INSERT INTO gpkg_contents (
                table_name, data_type, identifier, last_change, srs_id
            ) VALUES (
                'qtp_slope_units', 'features', 'qtp_slope_units',
                '2026-01-01T00:00:00.000Z', 100000
            );
            INSERT INTO gpkg_geometry_columns (
                table_name, column_name, geometry_type_name, srs_id, z, m
            ) VALUES ('qtp_slope_units', 'geom', 'MULTIPOLYGON', 100000, 0, 0);
            INSERT INTO gpkg_spatial_ref_sys (
                srs_name, srs_id, organization, organization_coordsys_id,
                definition, description
            ) VALUES (
                'QTP_Albers_Equal_Area', 100000, 'NONE', 100000,
                '{SOURCE_QTP_ALBERS_WKT}', ''
            );
            INSERT INTO qtp_slope_units (fid, geom, basin_id, su_id) VALUES
                (1, X'47500001', 'B1', 'B1_0001'),
                (2, X'47500002', 'B1', 'B1_0002'),
                (3, X'47500003', 'B2', 'B2_0001');
            """
        )


def write_prediction(path: Path, values: list[float | None], *, mismatch: bool = False) -> None:
    su_ids = ["B1_0001", "WRONG" if mismatch else "B1_0002", "B2_0001"]
    table = pa.table(
        {
            "fid": pa.array([1, 2, 3], type=pa.int64()),
            "su_id": pa.array(su_ids, type=pa.string()),
            "prediction": pa.array(values, type=pa.float32()),
        }
    )
    pq.write_table(table, path)


class SuPredictionMappingTest(unittest.TestCase):
    def test_domain_filter_removes_non_domain_geometries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_raw:
            source = Path(tmp_raw) / "QTP_SlopeUnits_predictions.gpkg"
            build_small_gpkg(source)
            with closing(sqlite3.connect(source)) as conn:
                conn.execute(
                    "CREATE TEMP TABLE permafrost_prediction_domain "
                    "(fid INTEGER PRIMARY KEY)"
                )
                conn.executemany(
                    "INSERT INTO permafrost_prediction_domain (fid) VALUES (?)",
                    [(1,), (3,)],
                )
                summary = filter_output_to_prediction_domain(
                    conn,
                    table_name="qtp_slope_units",
                    geometry_column="geom",
                    domain_table="permafrost_prediction_domain",
                    source_row_count=3,
                )
                rows = conn.execute(
                    "SELECT fid FROM qtp_slope_units ORDER BY fid"
                ).fetchall()
            self.assertEqual(summary["source_rows"], 3)
            self.assertEqual(summary["retained_rows"], 2)
            self.assertEqual(summary["excluded_rows"], 1)
            self.assertEqual(rows, [(1,), (3,)])

    def test_multiple_fields_are_attached_without_modifying_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_raw:
            tmp = Path(tmp_raw)
            source = tmp / "QTP_SlopeUnits.gpkg"
            output = tmp / "QTP_SlopeUnits_predictions.gpkg"
            result_128 = tmp / "predictions_bs128_2021.parquet"
            result_256 = tmp / "predictions_bs256_2022.parquet"
            build_small_gpkg(source)
            write_prediction(result_128, [0.1, None, 0.9])
            write_prediction(result_256, [0.2, 0.4, 0.8])
            source_before = source.read_bytes()

            summary = map_predictions(
                source_gpkg=source,
                output_gpkg=output,
                table_name="qtp_slope_units",
                specs=[
                    ResultSpec(result_128, "pred_bs128_2021", "prediction"),
                    ResultSpec(result_256, "pred_bs256_2022", "prediction"),
                ],
                fid_column="fid",
                su_id_column="su_id",
                skip_su_id_validation=False,
                require_complete=True,
                batch_rows=2,
                overwrite=False,
                restrict_to_prediction_domain=False,
            )

            self.assertEqual(source.read_bytes(), source_before)
            self.assertTrue(output.exists())
            self.assertFalse(summary["original_source_modified"])
            self.assertEqual(summary["source"]["row_count"], 3)
            self.assertEqual(summary["prediction_fields"], ["pred_bs128_2021", "pred_bs256_2022"])
            self.assertTrue(summary["arcgis_crs"]["enabled"])
            self.assertTrue(summary["arcgis_crs"]["applied"])
            with closing(sqlite3.connect(source)) as conn:
                source_columns = [row[1] for row in conn.execute("PRAGMA table_info(qtp_slope_units)")]
                source_definition = conn.execute(
                    "SELECT definition FROM gpkg_spatial_ref_sys WHERE srs_id = 100000"
                ).fetchone()[0]
            self.assertNotIn("pred_bs128_2021", source_columns)
            self.assertEqual(source_definition, SOURCE_QTP_ALBERS_WKT)
            with closing(sqlite3.connect(output)) as conn:
                output_columns = [row[1] for row in conn.execute("PRAGMA table_info(qtp_slope_units)")]
                output_definition = conn.execute(
                    "SELECT definition FROM gpkg_spatial_ref_sys WHERE srs_id = 100000"
                ).fetchone()[0]
                rows = conn.execute(
                    "SELECT fid, hex(geom), pred_bs128_2021, pred_bs256_2022 "
                    "FROM qtp_slope_units ORDER BY fid"
                ).fetchall()
            self.assertIn("pred_bs128_2021", output_columns)
            self.assertIn("pred_bs256_2022", output_columns)
            self.assertEqual(output_definition, ARCGIS_QTP_ALBERS_WKT)
            self.assertEqual([row[0] for row in rows], [1, 2, 3])
            self.assertEqual([row[1] for row in rows], ["47500001", "47500002", "47500003"])
            self.assertAlmostEqual(rows[0][2], 0.1, places=6)
            self.assertIsNone(rows[1][2])
            self.assertAlmostEqual(rows[2][2], 0.9, places=6)
            self.assertAlmostEqual(rows[1][3], 0.4, places=6)

    def test_su_id_mismatch_does_not_publish_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_raw:
            tmp = Path(tmp_raw)
            source = tmp / "QTP_SlopeUnits.gpkg"
            output = tmp / "QTP_SlopeUnits_predictions.gpkg"
            result = tmp / "predictions_bs128_2021.parquet"
            build_small_gpkg(source)
            write_prediction(result, [0.1, 0.5, 0.9], mismatch=True)
            source_before = source.read_bytes()

            with self.assertRaises(MappingError):
                map_predictions(
                    source_gpkg=source,
                    output_gpkg=output,
                    table_name="qtp_slope_units",
                    specs=[ResultSpec(result, "pred_bs128_2021", "prediction")],
                    fid_column="fid",
                    su_id_column="su_id",
                    skip_su_id_validation=False,
                    require_complete=True,
                    batch_rows=2,
                    overwrite=False,
                    restrict_to_prediction_domain=False,
                )

            self.assertFalse(output.exists())
            self.assertEqual(source.read_bytes(), source_before)
            self.assertEqual(list(tmp.glob(".*.tmp")), [])

    def test_prediction_nodata_sentinel_is_written_as_null(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_raw:
            tmp = Path(tmp_raw)
            source = tmp / "QTP_SlopeUnits.gpkg"
            output = tmp / "QTP_SlopeUnits_predictions.gpkg"
            result = tmp / "predictions_bs128_2021.parquet"
            build_small_gpkg(source)
            write_prediction(result, [-9999.0, 0.5, None])

            map_predictions(
                source_gpkg=source,
                output_gpkg=output,
                table_name="qtp_slope_units",
                specs=[ResultSpec(result, "pred_bs128_2021", "prediction")],
                fid_column="fid",
                su_id_column="su_id",
                skip_su_id_validation=False,
                require_complete=True,
                batch_rows=2,
                overwrite=False,
                restrict_to_prediction_domain=False,
            )

            with closing(sqlite3.connect(output)) as conn:
                rows = conn.execute(
                    "SELECT pred_bs128_2021 FROM qtp_slope_units ORDER BY fid"
                ).fetchall()
            self.assertIsNone(rows[0][0])
            self.assertAlmostEqual(rows[1][0], 0.5, places=6)
            self.assertIsNone(rows[2][0])

    def test_result_field_is_inferred_from_batch_size_and_year(self) -> None:
        spec = parse_result_spec(
            "predictions_dbscan_bs1024_2022.parquet",
            default_value_column="prediction",
        )
        self.assertEqual(spec.output_field, "pred_bs1024_2022")
        self.assertEqual(spec.value_column, "prediction")


if __name__ == "__main__":
    unittest.main()
