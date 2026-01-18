import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def load_yaml(rel_path: str) -> dict:
    path = ROOT / rel_path
    return yaml.safe_load(path.read_text())


def get_inputs(data: dict) -> dict:
    return data["on"]["execute"]["inputs"]


def get_run_service_step(data: dict) -> dict:
    steps = data["jobs"]["run_service"]["steps"]
    for step in steps:
        if step.get("name") == "Run Service":
            return step
    raise AssertionError("Run Service step not found")

def get_step_by_name(data: dict, job_name: str, step_name: str) -> dict:
    steps = data["jobs"][job_name]["steps"]
    for step in steps:
        if step.get("name") == step_name:
            return step
    raise AssertionError(f"{step_name} step not found in {job_name}")


def option_values(options):
    if options and isinstance(options[0], dict):
        return [opt.get("value") for opt in options]
    return options


class TestWorkflowConfigs(unittest.TestCase):
    def test_yaml_parse(self):
        for path in ("workflow.yaml", "yamls/hsp.yaml", "yamls/emed.yaml"):
            data = load_yaml(path)
            self.assertIsInstance(data, dict)
            self.assertIn("jobs", data)
            self.assertIn("on", data)

    def test_vllm_attention_backend_options(self):
        for path in ("workflow.yaml", "yamls/hsp.yaml", "yamls/emed.yaml"):
            data = load_yaml(path)
            inputs = get_inputs(data)
            backend = inputs["advanced_settings"]["items"]["vllm_attention_backend"]
            options = backend["options"]
            self.assertTrue(options, f"{path} vllm_attention_backend options missing")
            self.assertTrue(
                all(isinstance(opt, str) for opt in options),
                f"{path} vllm_attention_backend options must be strings",
            )

    def test_embedding_source_has_bucket(self):
        for path in ("workflow.yaml", "yamls/hsp.yaml", "yamls/emed.yaml"):
            data = load_yaml(path)
            inputs = get_inputs(data)
            values = option_values(inputs["rag"]["items"]["embedding_model_source"]["options"])
            self.assertIn("bucket", values, f"{path} embedding_model_source missing bucket")

    def test_container_source_options(self):
        for path in ("workflow.yaml", "yamls/hsp.yaml", "yamls/emed.yaml"):
            data = load_yaml(path)
            inputs = get_inputs(data)
            values = option_values(inputs["container"]["items"]["source"]["options"])
            for expected in ("lfs", "path", "pull", "build"):
                self.assertIn(expected, values, f"{path} container.source missing {expected}")

    def test_model_source_options(self):
        for path in ("workflow.yaml", "yamls/hsp.yaml", "yamls/emed.yaml"):
            data = load_yaml(path)
            inputs = get_inputs(data)
            values = option_values(inputs["model"]["items"]["source"]["options"])
            for expected in ("local", "huggingface"):
                self.assertIn(expected, values, f"{path} model.source missing {expected}")

    def test_runmode_options(self):
        for path in ("workflow.yaml", "yamls/hsp.yaml", "yamls/emed.yaml"):
            data = load_yaml(path)
            inputs = get_inputs(data)
            values = option_values(inputs["runmode"]["options"])
            for expected in ("singularity", "docker"):
                self.assertIn(expected, values, f"{path} runmode missing {expected}")

    def test_prepare_containers_steps(self):
        for path in ("workflow.yaml", "yamls/hsp.yaml", "yamls/emed.yaml"):
            data = load_yaml(path)
            step_ids = {step.get("id") for step in data["jobs"]["prepare_containers"]["steps"]}
            for expected in ("pull_containers", "pull_containers_lfs", "link_containers", "build_containers"):
                self.assertIn(expected, step_ids, f"{path} missing {expected} step")

    def test_prepare_embedding_model_includes_bucket(self):
        for path in ("workflow.yaml", "yamls/hsp.yaml", "yamls/emed.yaml"):
            data = load_yaml(path)
            cond = data["jobs"]["prepare_embedding_model"]["if"]
            self.assertIn("bucket", cond, f"{path} prepare_embedding_model missing bucket in if")

    def test_run_service_uses_job_runner(self):
        for path in ("workflow.yaml", "yamls/hsp.yaml", "yamls/emed.yaml"):
            data = load_yaml(path)
            run_step = get_run_service_step(data)
            self.assertEqual(run_step["uses"], "marketplace/job_runner/v4.0")

    def test_create_env_sets_embedding_cache_dir(self):
        for path in ("workflow.yaml", "yamls/hsp.yaml", "yamls/emed.yaml"):
            data = load_yaml(path)
            step = get_step_by_name(data, "setup", "Create Environment File")
            run_script = step.get("run", "")
            self.assertIn("EMBEDDING_CACHE_DIR", run_script, f"{path} missing EMBEDDING_CACHE_DIR in env")

    def test_hsp_defaults(self):
        data = load_yaml("yamls/hsp.yaml")
        inputs = get_inputs(data)
        self.assertTrue(inputs["scheduler"]["items"]["enabled"]["default"])
        self.assertEqual(inputs["vllm"]["items"]["num_gpus"]["default"], "4")
        self.assertEqual(inputs["container"]["items"]["source"]["default"], "pull")
        self.assertEqual(
            inputs["model"]["items"]["hf_model_id"]["default"],
            "nvidia/Llama-3_3-Nemotron-Super-49B-v1_5",
        )
        self.assertEqual(inputs["model"]["items"]["cache_dir"]["default"], "${WORKDIR}")
        self.assertEqual(inputs["rag"]["items"]["embedding_model_source"]["default"], "bucket")
        self.assertEqual(inputs["rag"]["items"]["embedding_model_cache_dir"]["default"], "${WORKDIR}")
        self.assertEqual(inputs["scheduler"]["items"]["slurm"]["items"]["gres"]["default"], "gpu:4")
        self.assertIn(
            "##SBATCH --constraint=mla",
            inputs["scheduler"]["items"]["slurm"]["items"]["scheduler_directives"]["default"],
        )

    def test_wait_server_localhost_fallback(self):
        for path in ("workflow.yaml", "yamls/hsp.yaml"):
            data = load_yaml(path)
            step = get_step_by_name(data, "create_session", "Wait for Server")
            run_script = step.get("run", "")
            self.assertIn("local_host", run_script, f"{path} missing localhost fallback in wait_server")

    def test_emed_hpc4_overrides(self):
        data = load_yaml("yamls/emed.yaml")
        inputs = get_inputs(data)
        slurm = inputs["scheduler"]["items"]["slurm"]["items"]
        self.assertIn("cluster", slurm)
        self.assertIn("partition_hpc4", slurm)
        self.assertIn("gres_gpu_hpc4", slurm)
        cluster_opts = slurm["cluster"]["options"]
        if isinstance(cluster_opts, list) and cluster_opts and isinstance(cluster_opts[0], dict):
            values = [opt.get("value") for opt in cluster_opts]
        else:
            values = cluster_opts
        self.assertIn("hpc4", values)

        self.assertIn("derive_scheduler", data["jobs"])
        self.assertIn("derive_scheduler", data["jobs"]["run_service"]["needs"])
        run_step = get_run_service_step(data)
        slurm_with = run_step["with"]["slurm"]
        self.assertEqual(
            slurm_with["partition"],
            "${{ needs.derive_scheduler.outputs.slurm_partition }}",
        )
        self.assertEqual(
            slurm_with["gres"],
            "${{ needs.derive_scheduler.outputs.slurm_gres }}",
        )
        self.assertIn(
            "${{ needs.derive_scheduler.outputs.slurm_directives }}",
            slurm_with["scheduler_directives"],
        )


if __name__ == "__main__":
    unittest.main()
