CONFIG ?= ../../examples/simulation_pos_smoke.yaml

.PHONY: all smoke oracle phase2 train virtual evaluate figures slurm-command

all:
	conda run -n env_2502 python -m experiments.simulation_pos.run_pipeline --config $(CONFIG)

smoke:
	conda run -n env_2502 python -m experiments.simulation_pos.run_pipeline --config $(CONFIG) --smoke

oracle:
	conda run -n env_2502 python -m experiments.simulation_pos.run_pipeline --config $(CONFIG) --stage oracle

phase2:
	conda run -n env_2502 python -m experiments.simulation_pos.run_pipeline --config $(CONFIG) --stage phase2

train:
	conda run -n env_2502 python -m experiments.simulation_pos.run_pipeline --config $(CONFIG) --stage train

virtual:
	conda run -n env_2502 python -m experiments.simulation_pos.run_pipeline --config $(CONFIG) --stage virtual

evaluate:
	conda run -n env_2502 python -m experiments.simulation_pos.run_pipeline --config $(CONFIG) --stage evaluate

figures:
	conda run -n env_2502 python -m experiments.simulation_pos.run_pipeline --config $(CONFIG) --stage figures

slurm-command:
	@echo "Create a site-specific Slurm script for $(CONFIG), then submit it with sbatch."
