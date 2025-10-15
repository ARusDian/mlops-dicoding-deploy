"""Pipeline TFX untuk proyek 'arusdian'.

File ini mendefinisikan rangkaian komponen TFX (ExampleGen, StatisticsGen,
SchemaGen, ExampleValidator, Transform, Tuner, Trainer, Evaluator, Pusher)
serta fungsi utilitas untuk menjalankan pipeline menggunakan BeamDagRunner.

Struktur direktori yang dipakai:
- data/                    : sumber data (CSV)
- arusdian-pipeline/       : kode pipeline (file ini)
- arusdian-pipeline/modules: modul transform, trainer, tuner
- arusdian-pipeline/artifacts: artifact TFX
- serving_model/arusdian   : direktori hasil model yang didorong (push)
- arusdian-pipeline/metadata/metadata.db : metadata mlmd
"""

import os
from typing import List

import tensorflow_model_analysis as tfma

from tfx.components import (
    CsvExampleGen,
    StatisticsGen,
    SchemaGen,
    ExampleValidator,
    Transform,
    Tuner,
    Trainer,
    Evaluator,
    Pusher,
)
from tfx.proto import example_gen_pb2, trainer_pb2, pusher_pb2
from tfx.dsl.components.common.resolver import Resolver
from tfx.dsl.input_resolution.strategies.latest_blessed_model_strategy import (
    LatestBlessedModelStrategy,
)
from tfx.types import Channel
from tfx.types.standard_artifacts import Model, ModelBlessing

from tfx.orchestration import metadata, pipeline
from tfx.orchestration.beam.beam_dag_runner import BeamDagRunner


# ---------------- CONFIG ----------------
PIPELINE_NAME = "arusdian"
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# lokasi data barumu: submission-2/data/heart.csv
DATA_DIR = os.path.join(PROJECT_ROOT, "..", "data")

PIPELINE_ROOT = os.path.join(PROJECT_ROOT, "artifacts")
SERVING_MODEL_DIR = os.path.join(PROJECT_ROOT, "..", "serving_model", PIPELINE_NAME)
MODULES_DIR = os.path.join(PROJECT_ROOT, "modules")
TRANSFORM_MODULE_FILE = os.path.join(MODULES_DIR, "heart_transform.py")
TUNER_MODULE_FILE = os.path.join(MODULES_DIR, "heart_tuner.py")
TRAINER_MODULE_FILE = os.path.join(MODULES_DIR, "heart_trainer.py")
METADATA_PATH = os.path.join(PROJECT_ROOT, "metadata", "metadata.db")

# Beam args (sesuaikan jika dipakai di cloud runner)
BEAM_PIPELINE_ARGS: List[str] = [
    "--direct_num_workers=0",
]


def create_pipeline():
    """Membangun daftar komponen TFX untuk pipeline arusdian.

    Komponen yang dibuat:
      - ExampleGen: membaca CSV dan membagi train/eval (80:20 via hash)
      - StatisticsGen: menghitung statistik fitur
      - SchemaGen: menginfer schema dari statistik
      - ExampleValidator: validasi contoh terhadap schema
      - Transform: feature engineering via TF Transform (module_file)
      - Tuner: pencarian hyperparameter
      - Trainer: melatih model (menggunakan hasil transform & hparams)
      - Resolver: mengambil model blessed terbaru (sebagai baseline)
      - Evaluator: evaluasi via TFMA + threshold metric
      - Pusher: mendorong model ke direktori serving jika blessed

    Returns:
        list: Urutan komponen TFX yang akan dieksekusi oleh runner.
    """
    # ---------- ExampleGen (hash split 80:20 dari satu file CSV) ----------
    output_cfg = example_gen_pb2.Output(
        split_config=example_gen_pb2.SplitConfig(
            splits=[
                example_gen_pb2.SplitConfig.Split(name="train", hash_buckets=8),
                example_gen_pb2.SplitConfig.Split(name="eval", hash_buckets=2),
            ]
        )
    )
    example_gen = CsvExampleGen(input_base=DATA_DIR, output_config=output_cfg)

    # ---------- StatisticsGen ----------
    statistics_gen = StatisticsGen(examples=example_gen.outputs["examples"])

    # ---------- SchemaGen ----------
    schema_gen = SchemaGen(
        statistics=statistics_gen.outputs["statistics"], infer_feature_shape=True
    )

    # ---------- ExampleValidator ----------
    example_validator = ExampleValidator(
        statistics=statistics_gen.outputs["statistics"],
        schema=schema_gen.outputs["schema"],
    )

    # ---------- Transform ----------
    transform = Transform(
        examples=example_gen.outputs["examples"],
        schema=schema_gen.outputs["schema"],
        module_file=TRANSFORM_MODULE_FILE,
    )

    # ---------- Tuner ----------
    tuner = Tuner(
        module_file=TUNER_MODULE_FILE,
        examples=transform.outputs["transformed_examples"],
        transform_graph=transform.outputs["transform_graph"],
        train_args=trainer_pb2.TrainArgs(splits=["train"], num_steps=400),
        eval_args=trainer_pb2.EvalArgs(splits=["eval"], num_steps=200),
    )

    # ---------- Trainer ----------
    trainer = Trainer(
        module_file=TRAINER_MODULE_FILE,
        custom_config={"transform_module_file": TRANSFORM_MODULE_FILE},
        examples=transform.outputs["transformed_examples"],
        transform_graph=transform.outputs["transform_graph"],
        schema=schema_gen.outputs["schema"],
        hyperparameters=tuner.outputs["best_hyperparameters"],
        train_args=trainer_pb2.TrainArgs(splits=["train"], num_steps=800),
        eval_args=trainer_pb2.EvalArgs(splits=["eval"], num_steps=200),
    )

    # ---------- Resolver (latest blessed model) ----------
    model_resolver = Resolver(
        strategy_class=LatestBlessedModelStrategy,
        model=Channel(type=Model),
        model_blessing=Channel(type=ModelBlessing),
    ).with_id("latest_blessed_model_resolver")

    # ---------- Evaluator (TFMA) ----------
    eval_config = tfma.EvalConfig(
        model_specs=[tfma.ModelSpec(label_key="HeartDisease")],
        slicing_specs=[tfma.SlicingSpec()],
        metrics_specs=[
            tfma.MetricsSpec(
                metrics=[
                    tfma.MetricConfig(class_name="ExampleCount"),
                    tfma.MetricConfig(class_name="AUC"),
                    tfma.MetricConfig(class_name="FalsePositives"),
                    tfma.MetricConfig(class_name="TruePositives"),
                    tfma.MetricConfig(class_name="FalseNegatives"),
                    tfma.MetricConfig(class_name="TrueNegatives"),
                    tfma.MetricConfig(
                        class_name="BinaryAccuracy",
                        threshold=tfma.MetricThreshold(
                            value_threshold=tfma.GenericValueThreshold(
                                lower_bound={"value": 0.5}
                            ),
                            change_threshold=tfma.GenericChangeThreshold(
                                direction=tfma.MetricDirection.HIGHER_IS_BETTER,
                                absolute={"value": 0.0001},
                            ),
                        ),
                    ),
                ]
            )
        ],
    )
    evaluator = Evaluator(
        examples=example_gen.outputs["examples"],
        model=trainer.outputs["model"],
        baseline_model=model_resolver.outputs["model"],
        eval_config=eval_config,
    )

    # ---------- Pusher ----------
    pusher = Pusher(
        model=trainer.outputs["model"],
        model_blessing=evaluator.outputs["blessing"],
        push_destination=pusher_pb2.PushDestination(
            filesystem=pusher_pb2.PushDestination.Filesystem(
                base_directory=SERVING_MODEL_DIR
            )
        ),
    )

    return [
        example_gen,
        statistics_gen,
        schema_gen,
        example_validator,
        transform,
        tuner,
        trainer,
        model_resolver,
        evaluator,
        pusher,
    ]


def run():
    """Menyiapkan objek Pipeline dan mengeksekusinya dengan BeamDagRunner."""
    pl = pipeline.Pipeline(
        pipeline_name=PIPELINE_NAME,
        pipeline_root=PIPELINE_ROOT,
        metadata_connection_config=metadata.sqlite_metadata_connection_config(
            METADATA_PATH
        ),
        components=create_pipeline(),
        enable_cache=True,
        beam_pipeline_args=BEAM_PIPELINE_ARGS,
    )
    BeamDagRunner().run(pl)


if __name__ == "__main__":
    os.makedirs(PIPELINE_ROOT, exist_ok=True)
    os.makedirs(os.path.dirname(METADATA_PATH), exist_ok=True)
    os.makedirs(SERVING_MODEL_DIR, exist_ok=True)
    run()
