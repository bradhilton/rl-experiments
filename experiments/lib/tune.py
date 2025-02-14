import asyncio
import glob
from lib.mlp_head_checkpointer import MLPHeadCheckpointer
from lib.pack import PackedDataset, PackedTensors, packed_tensors_to_dir
from lib.recipe import ComponentConfig, recipe_main, TuneRecipeConfig
from omegaconf import OmegaConf
import os
import re
import sys
from torchtune.modules import TransformerDecoder
from torchtune.training import cleanup_before_training, FullModelHFCheckpointer
from torchtune.training.metric_logging import DiskLogger
import tqdm
from typing import Any, Literal, IO


Verbosity = Literal[0, 1, 2]


async def tune(
    base_model: str,
    output_dir: str,
    packed_tensors: PackedTensors,
    model: TransformerDecoder,
    model_type: str,
    config: TuneRecipeConfig = TuneRecipeConfig(),
    in_process: bool = False,
    verbosity: Verbosity = 2,
) -> None:
    process = await asyncio.create_subprocess_shell(
        f"HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download {base_model}",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, _ = await process.communicate()
    base_checkpoint_dir = stdout.decode().strip()

    config.checkpointer = get_checkpointer_config(
        checkpoint_dir=base_checkpoint_dir,
        output_dir=output_dir,
        tune_model_type=model_type,
    )
    config.reference_checkpointer = get_checkpointer_config(
        checkpoint_dir=base_checkpoint_dir,
        output_dir=output_dir,
        tune_model_type=model_type,
    )
    config.metric_logger = ComponentConfig(DiskLogger, log_dir=f"{output_dir}/logs")
    config.model = ComponentConfig(model)
    disk_packed_tensors = packed_tensors_to_dir(packed_tensors, f"{output_dir}/tensors")
    config.dataset = ComponentConfig(
        PackedDataset,
        **disk_packed_tensors,
    )
    config.seed = 42
    dict_config = config.dict_config()
    print(OmegaConf.to_yaml(dict_config))
    OmegaConf.save(dict_config, f"{output_dir}/config.yaml")
    if in_process:
        cleanup_before_training()
        recipe_main(config)
    else:
        await tune_run(
            config_path=f"{output_dir}/config.yaml",
            total=disk_packed_tensors["num_sequences"],
            verbosity=verbosity,
            tune_run_env={"CUDA_LAUNCH_BLOCKING": "1"},
        )


def get_checkpointer_config(
    checkpoint_dir: str,
    output_dir: str,
    tune_model_type: str,
    checkpoint_files: list[str] | None = None,
    mlp_head_checkpointer: bool = False,
    output_subdir: str = "",
) -> ComponentConfig[FullModelHFCheckpointer]:
    return ComponentConfig(
        MLPHeadCheckpointer if mlp_head_checkpointer else FullModelHFCheckpointer,
        checkpoint_dir=checkpoint_dir,
        checkpoint_files=checkpoint_files
        or [
            file
            for ext in ["safetensors", "pt", "ckpt", "bin", "pth"]
            for file in glob.glob(f"{checkpoint_dir}/*.{ext}")
            if not file.endswith("mlp_head.pt")
        ],
        recipe_checkpoint=None,
        output_dir=output_dir + output_subdir,
        model_type=tune_model_type,
    )


async def tune_run(
    config_path: str,
    total: int,
    verbosity: Verbosity = 2,
    torchrun_kwargs: dict[str, Any] | None = None,
    tune_run_env: dict[str, str] | None = None,
) -> None:
    args = [
        "tune",
        "run",
        *[
            f"--{key.replace('_', '-')}{f'={value}' if value is not True else ''}"
            for key, value in (torchrun_kwargs or {}).items()
        ],
        "lib.recipe.TuneRecipe",
        "--config",
        config_path,
    ]
    if verbosity > 0:
        print(f"$ {' '.join(args)}")
    process = await asyncio.create_subprocess_exec(
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env={**os.environ, **(tune_run_env or {})},
    )
    if verbosity == 1:
        pbar = tqdm.tqdm(total=total)
    else:
        pbar = None

    async def log_output(stream: asyncio.StreamReader, io: IO[str]) -> None:
        output = ""
        while True:
            try:
                chunk = await stream.read(4096)
                if not chunk:
                    break
                output += chunk.decode()
                if verbosity > 1:
                    io.write(output)
                    io.flush()
                    output = ""
                elif verbosity == 1:
                    output = output.split("\n")[-1]
                    if pbar:
                        pbar_start = re.compile(r"(\d+)\|(\d+)\|Loss: ([\d.]+):")
                        if match := pbar_start.search(output):
                            epoch, step, loss = match.groups()
                            pbar.update(int(step) - pbar.n)
                            pbar.set_description(f"{epoch}|{step}|Loss: {loss}")
                        metrics = {
                            key: value
                            for key, value in re.findall(r"(\w+)=([\d.-]+)", output)
                        }
                        if metrics:
                            pbar.set_postfix(**metrics)
                            output = ""
                    else:
                        pbar_regex = re.compile(
                            r"\[(?:\d+:)?\d+:\d+<(?:\d+:)?\d+:\d+.*\]"
                        )
                        if pbar_regex.search(output):
                            io.write(output)
                            io.flush()
                            output = ""
            except Exception:
                break

    tasks = []
    if process.stdout:
        tasks.append(asyncio.create_task(log_output(process.stdout, sys.stdout)))
    if process.stderr:
        tasks.append(asyncio.create_task(log_output(process.stderr, sys.stderr)))
    try:
        _ = await asyncio.gather(*tasks)
    except asyncio.CancelledError:
        process.kill()
    if pbar:
        pbar.close()
