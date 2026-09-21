"""Meaningful private-job and UI regression checks; no changes to study labels."""
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch, Mock
import zipfile

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "webapp"))
import nar_jobs as jobs


class PrivateJobTests(unittest.TestCase):
    def test_pending_health_is_read_only_and_detects_worker_token(self):
        import nar_recovery
        import time
        with tempfile.TemporaryDirectory() as temp, patch.object(jobs,'JOBS',Path(temp)):
            token=jobs.submit('demo','rice',launch=False)
            state=jobs.read_state(token);state['submitted_unix']=time.time()-120
            jobs.save_state(jobs.job_path(token),state)
            before=(jobs.job_path(token)/'state.json').read_bytes()
            with patch.object(nar_recovery.psutil,'process_iter',return_value=[]):
                self.assertEqual(nar_recovery.execution_health(token),'no_worker_detected')
            proc=Mock();proc.pid=987654;proc.name.return_value='python.exe'
            proc.cmdline.return_value=['python.exe',str(ROOT/'webapp/nar_jobs.py'),token]
            with patch.object(nar_recovery.psutil,'process_iter',return_value=[proc]):
                self.assertEqual(nar_recovery.execution_health(token),'active')
            proc.cmdline.side_effect=nar_recovery.psutil.AccessDenied(proc.pid)
            with patch.object(nar_recovery.psutil,'process_iter',return_value=[proc]):
                self.assertEqual(nar_recovery.execution_health(token),'unknown')
            self.assertEqual(before,(jobs.job_path(token)/'state.json').read_bytes())

    def test_queue_cap_rejects_without_creating_another_job(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(jobs,'JOBS',Path(temp)):
            for _ in range(3):jobs.submit('demo','rice',launch=False)
            before=set(Path(temp).iterdir())
            with self.assertRaisesRegex(ValueError,'queue is full'):
                jobs.submit('demo','rice',launch=False)
            self.assertEqual(before,set(Path(temp).iterdir()))

    def test_retention_removes_raw_files_then_terminal_job_only(self):
        import time
        with tempfile.TemporaryDirectory() as temp, patch.object(jobs,'JOBS',Path(temp)):
            token=jobs.submit('raw','rice',{'protein':b'>query\nACDEFGHIK\n'},launch=False)
            path=jobs.job_path(token)
            (path/'embeddings').mkdir()
            (path/'embeddings'/'private.npy').write_bytes(b'x')
            (path/'results.tsv').write_text('gene_id\tscore\nquery\t0.5\n')
            state=jobs.read_state(token);state.update(state='complete',completed_unix=time.time()-jobs.INPUT_RETENTION_SECONDS-1)
            jobs.save_state(path,state)
            self.assertEqual(jobs.cleanup_expired_jobs()['inputs'],1)
            self.assertFalse((path/'protein.fasta').exists())
            self.assertFalse((path/'embeddings').exists())
            self.assertTrue((path/'results.tsv').exists())
            state=jobs.read_state(token);state['completed_unix']=time.time()-jobs.JOB_RETENTION_SECONDS-1
            jobs.save_state(path,state)
            self.assertEqual(jobs.cleanup_expired_jobs()['jobs'],1)
            self.assertFalse(path.exists())

    def test_traversal_and_unknown_links(self):
        for token in ["../state.json", "a" * 42, "a" * 44, "a/" * 22]:
            with self.assertRaises(ValueError):
                jobs.job_path(token)

    def test_input_rejects_duplicate_and_invalid_sequences(self):
        for data in [b"no fasta", b">x\nAB!\n", b">x\nACD\n>x\nACD\n"]:
            with self.assertRaises(ValueError):
                jobs.validate_protein(data)
        self.assertEqual(jobs.validate_protein(b">example\nACDEFGHIK\n"), 1)

    def test_private_state_failure_and_escaped_report(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(jobs, "JOBS", Path(temp)):
            token = jobs.submit("demo", "rice", launch=False)
            other = jobs.submit("demo", "rice", launch=False)
            self.assertEqual(len(jobs.read_state(token)["input_sha256"]["bundled_feature_matrix"]), 64)
            self.assertNotEqual(token, other)
            self.assertEqual(len(token), 43)
            frame = pd.DataFrame({"gene_id": ["<script>evil</script>", "safe"], "score": [0.8, 0.2]})
            with patch.object(jobs, "demo_prediction", return_value=frame):
                jobs.run_job(token)
            self.assertEqual(jobs.read_state(token)["state"], "complete")
            self.assertEqual(jobs.read_state(other)["state"], "queued")
            report = (jobs.job_path(token) / "report.html").read_text(encoding="utf-8")
            self.assertNotIn("<script>", report)
            self.assertIn("&lt;script&gt;", report)
            with zipfile.ZipFile(jobs.job_path(token) / "report.zip") as archive:
                self.assertEqual(set(archive.namelist()), {"results.tsv", "results.csv", "report.html", "provenance.json", "score_distribution.svg"})
            with patch.object(jobs, "demo_prediction", side_effect=RuntimeError("private/path")):
                jobs.run_job(other)
            self.assertEqual(jobs.read_state(other)["state"], "failed")
            self.assertNotIn("private/path", json.dumps(jobs.read_state(other)))

    def test_ui_landing_and_result_route(self):
        from streamlit.testing.v1 import AppTest
        app = AppTest.from_file(str(ROOT / "webapp/nar_app.py"), default_timeout=30).run()
        self.assertEqual(len(app.exception), 0)
        self.assertTrue(any("PlantEGP" in x.value for x in app.markdown))
        app.query_params["job"] = "../../invalid"
        app.run()
        self.assertEqual(len(app.exception), 0)
        self.assertTrue(any("invalid" in x.value for x in app.error))

    def test_annotation_rejection_before_job_creation(self):
        from nar_inputs import validate_annotations
        base={"protein":b">query\nACDEFGHIK\n"}
        for data in [b"gene_id\tgo_id\nwrong\tGO:0008150\n",b"gene_id\tgo_id\nquery\tinvalid\n"]:
            with self.assertRaises(ValueError):validate_annotations({**base,"go":data})
        result=validate_annotations({**base,"go":b"gene_id\tgo_id\nquery\tGO:0008150\n"})
        self.assertEqual(result['go']['matched_query_genes'],1)

    def test_recovery_preserves_original_and_refuses_active_child(self):
        import nar_recovery
        with tempfile.TemporaryDirectory() as temp, patch.object(jobs,'JOBS',Path(temp)):
            token=jobs.submit('demo','rice',launch=False)
            state=jobs.read_state(token);state['state']='running';jobs.save_state(jobs.job_path(token),state)
            with patch.object(nar_recovery,'active_processes',return_value=[123]):
                with self.assertRaises(ValueError):nar_recovery.clone_for_recovery(token)
            with patch.object(nar_recovery,'active_processes',return_value=[]):
                new=nar_recovery.clone_for_recovery(token)
            self.assertEqual(jobs.read_state(token)['state'],'running')
            self.assertEqual(jobs.read_state(new)['recovery_of'],token)
            self.assertEqual(jobs.read_state(new)['state'],'queued')

    def test_partial_coverage_and_empty_annotation(self):
        from nar_inputs import validate_annotations
        base={'protein':b'>a\nACDEFG\n>b\nACDEFG\n'}
        with self.assertRaises(ValueError):validate_annotations({**base,'go':b''})
        result=validate_annotations({**base,'go':b'gene_id\tgo_id\na\tGO:0008150\n'})
        self.assertEqual(result['go']['query_coverage_fraction'],0.5)
        self.assertEqual(result['go']['unmatched_query_genes'],['b'])

    def test_optional_search_failure_preserves_prediction(self):
        import nar_evidence
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);ref=root/'reference';ref.mkdir()
            (ref/'manifest.json').write_text(json.dumps({'diamond_executable':'missing_executable'}),encoding='utf-8')
            (root/'protein.fasta').write_text('>query\nACDEFGHIK\n',encoding='utf-8')
            frame=pd.DataFrame({'gene_id':['query'],'score':[.65]})
            with patch.dict('os.environ',{'PLANT_EG_REFERENCE':str(ref)}),patch.object(nar_evidence.subprocess,'run',side_effect=OSError('private/path')):
                out=nar_evidence.attach_homology(frame,root,'rice')
            self.assertEqual(out.score.iloc[0],.65)
            self.assertEqual(out.homology_search_status.iloc[0],'search_failed_prediction_preserved')


if __name__ == "__main__":
    unittest.main()
