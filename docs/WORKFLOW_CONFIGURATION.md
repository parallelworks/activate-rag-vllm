# Workflow Configuration Guide

This guide explains how to customize the ACTIVATE workflow YAML for different environments and use cases.

## Workflow Structure

The `workflow.yaml` file defines how the service deploys on the ACTIVATE platform:

```yaml
permissions:         # API permissions
sessions:            # Session configuration (ports, OpenAI access)
jobs:                # Job definitions (steps to execute)
app:                 # Application metadata
form:                # User input form definition
```

## Form Configuration

### Form Groups

Inputs are organized into logical groups:

```yaml
form:
  resource:          # Cluster selection
  scheduler:         # Job scheduler settings (SLURM/PBS/SSH)
  runmode:           # Container runtime (Apptainer/Singularity/Docker)
  model:             # Model configuration group
  vllm_extra_args:   # vLLM server options
  container:         # Container pull settings
  advanced_settings: # Advanced options
```

### Input Types

| Type | Description | Example |
|------|-------------|---------|
| `string` | Text input | Model path |
| `password` | Hidden input | HF token |
| `dropdown` | Select from options | Scheduler type |
| `boolean` | True/false toggle | Build containers |
| `number` | Numeric input | Port numbers |
| `group` | Nested inputs | Model config |

### Conditional Fields

Use `hidden` and `ignore` to show/hide fields based on other inputs:

```yaml
# Only show when model source is HuggingFace
hf_model_id:
  type: string
  label: HuggingFace Model ID
  hidden: ${{ inputs.model.source != 'huggingface' }}
  ignore: ${{ inputs.model.source != 'huggingface' }}
```

Common patterns:
- `hidden: ${{ inputs.X != 'value' }}` - Hide unless X equals value
- `ignore: ${{ inputs.X != 'value' }}` - Don't include in submission
- `hidden: ${{ inputs.runmode != 'singularity' }}` - Apptainer-specific options

## RAG Configuration

Embedding model options are configured under the `rag` group:

```yaml
rag:
  items:
    embedding_model_source:
      type: dropdown
      default: huggingface
      options:
        - value: local
        - value: huggingface
        - value: bucket
```

The HSP preset (`yamls/hsp.yaml`) defaults `embedding_model_source` to `huggingface` so embeddings are cloned via git-lfs instead of pulling from a bucket on each run.

## Scheduler Configuration

### SLURM

```yaml
slurm:
  partition:
    type: string
    default: gpu
  constraint:
    type: string
    default: mla
    tooltip: Node constraint (e.g., 'mla' for A100 nodes)
  account:
    type: string
    optional: true
```

Job script generation:
```yaml
steps:
  - name: Submit SLURM Job
    if: ${{ inputs.scheduler.type == 'slurm' }}
    run: |
      sbatch <<EOF
      #!/bin/bash
      #SBATCH --job-name=vllm-rag
      #SBATCH --partition=${{ inputs.scheduler.slurm.partition }}
      #SBATCH --gres=gpu:4
      #SBATCH --constraint=${{ inputs.scheduler.slurm.constraint }}
      ...
      EOF
```

### PBS

```yaml
pbs:
  queue:
    type: string
    default: gpu
  select:
    type: string
    default: "1:ncpus=92:mpiprocs=1:ngpus=4"
```

Job script generation:
```yaml
- name: Submit PBS Job
  if: ${{ inputs.scheduler.type == 'pbs' }}
  run: |
    qsub <<EOF
    #!/bin/bash
    #PBS -N vllm-rag
    #PBS -q ${{ inputs.scheduler.pbs.queue }}
    #PBS -l select=${{ inputs.scheduler.pbs.select }}
    ...
    EOF
```

### SSH (Direct Execution)

```yaml
- name: Run via SSH
  if: ${{ inputs.scheduler.type == 'ssh' }}
  run: |
    cd ${{ inputs.rundir }}
    ./start_service.sh &
```

## Model Configuration

### Model Source Selection

```yaml
model:
  type: group
  label: Model Configuration
  items:
    source:
      type: dropdown
      label: Model Source
      default: local
      options:
        - value: local
          label: "📁 Local Path (pre-staged weights)"
        - value: huggingface
          label: "🤗 HuggingFace Clone (git-lfs)"
```

### Local Model Path

```yaml
    local_path:
      type: string
      label: Model Path
      placeholder: /path/to/model/weights
      default: /models/Llama-3_3-Nemotron-Super-49B-v1_5
      hidden: ${{ inputs.model.source != 'local' }}
      ignore: ${{ inputs.model.source != 'local' }}
```

### HuggingFace Clone Configuration

```yaml
    hf_model_id:
      type: string
      label: HuggingFace Model ID
      placeholder: nvidia/Llama-3_3-Nemotron-Super-49B-v1_5
      hidden: ${{ inputs.model.source != 'huggingface' }}
      ignore: ${{ inputs.model.source != 'huggingface' }}
    
    hf_token:
      type: password
      label: HuggingFace Token
      default: ${{ org.HF_TOKEN }}
      hidden: ${{ inputs.model.source != 'huggingface' }}
      ignore: ${{ inputs.model.source != 'huggingface' }}
    
    cache_dir:
      type: string
      label: Model Cache Directory
      default: ~/pw/models
      hidden: ${{ inputs.model.source != 'huggingface' }}
```

## Container Configuration

### Apptainer/Singularity Containers

```yaml
container:
  type: group
  label: Container Options
  hidden: ${{ inputs.runmode != 'singularity' }}
  items:
    pull:
      type: boolean
      label: Pull Pre-built Containers
      default: true
    
    bucket:
      type: string
      label: Container Bucket
      default: pw://mshaxted/codeassist
      hidden: ${{ inputs.container.pull != true }}
```

### Container Pull Step

```yaml
- name: Pull Apptainer Containers
  if: ${{ inputs.runmode == 'singularity' && inputs.container.pull == true }}
  run: |
    if [[ ! -f "vllm.sif" ]]; then
      pw bucket cp "${{ inputs.container.bucket }}/vllm.sif" ./
    fi
```

## Environment File Generation

The workflow creates `.run.env` with all configuration:

```yaml
- name: Create Environment File
  run: |
    cd ${{ inputs.rundir }}
    
    echo "export RUNMODE=${{ inputs.runmode }}" > .run.env
    echo "export RUNTYPE=${{ inputs.runtype }}" >> .run.env
    echo "export MODEL_NAME=${{ inputs.model.local_path }}" >> .run.env
    # ... additional variables
```

## Advanced Settings

### vLLM Attention Backend

```yaml
advanced_settings:
  type: group
  collapsed: true
  items:
    vllm_attention_backend:
      type: dropdown
      label: Attention Backend
      default: ""
      options:
        - value: ""
          label: Auto-detect
        - value: FLASH_ATTN
          label: Flash Attention
        - value: FLASHINFER
          label: FlashInfer
```

### Tiktoken Encodings

For offline tokenizer support:

```yaml
    tiktoken_encodings:
      type: boolean
      label: Include Tiktoken Encodings
      default: true
      tooltip: Required for offline tokenizer support
```

### Repository Branch

```yaml
    repository_branch:
      type: string
      label: Repository Branch
      default: main
      tooltip: Branch to checkout for deployment
```

## Job Steps

### Step Structure

```yaml
jobs:
  prepare_job_directory:
    ssh:
      remoteHost: ${{ inputs.resource.ip }}
    steps:
      - name: Step Name
        if: ${{ condition }}
        early-cancel: any-job-failed
        run: |
          # Shell commands
```

### Conditional Execution

```yaml
# Run only for Apptainer/Singularity mode
- name: Install Apptainer Compose
  if: ${{ inputs.runmode == 'singularity' }}

# Run only for HuggingFace models
- name: Download HuggingFace Model
  if: ${{ inputs.model.source == 'huggingface' }}

# Run only for SLURM scheduler
- name: Submit SLURM Job
  if: ${{ inputs.scheduler.type == 'slurm' }}
```

### Error Handling

```yaml
- name: Critical Step
  early-cancel: any-job-failed
  run: |
    set -e  # Exit on error
    ...
```

## HPC Presets

See `configs/hpc-presets.yaml` for environment-specific configurations:

```yaml
presets:
  navy-dsrc:
    slurm_partition: gpu
    slurm_constraint: mla
    container_bucket: pw://navy/containers
    
  afrl-dsrc:
    slurm_partition: gpuq
    pbs_queue: standard
```

## Customization Examples

### Adding a New Form Field

```yaml
form:
  my_custom_option:
    type: string
    label: My Custom Option
    default: "default_value"
    tooltip: Explanation for users
```

Use in job steps:
```yaml
run: |
  echo "Custom option: ${{ inputs.my_custom_option }}"
```

### Adding a New Scheduler

1. Add dropdown option:
```yaml
scheduler:
  type:
    options:
      - value: my_scheduler
        label: My Scheduler
```

2. Add configuration group:
```yaml
  my_scheduler:
    hidden: ${{ inputs.scheduler.type != 'my_scheduler' }}
    items:
      queue:
        type: string
        default: default
```

3. Add job step:
```yaml
- name: Submit My Scheduler Job
  if: ${{ inputs.scheduler.type == 'my_scheduler' }}
  run: |
    my-scheduler-submit ...
```

### Changing Default Model

```yaml
model:
  items:
    local_path:
      default: /shared/models/your-preferred-model
```

## Debugging Workflows

### Dry Run

Test workflow form without submitting:
1. Fill out the form in ACTIVATE
2. Check "Dry Run" if available
3. Review generated configuration

### Local Testing

Use `run_local.sh` to test the service logic:
```bash
./run_local.sh --dry-run --model /path/to/model
```

### Checking Generated Environment

After workflow runs, check `.run.env`:
```bash
cat $RUNDIR/.run.env
```

## Reference

### Variable Substitution

| Syntax | Description |
|--------|-------------|
| `${{ inputs.X }}` | Form input value |
| `${{ inputs.X.Y }}` | Nested group value |
| `${{ org.VAR }}` | Organization secret |
| `${{ env.VAR }}` | Environment variable |

### Useful Conditions

| Condition | Meaning |
|-----------|---------|
| `${{ inputs.X == 'value' }}` | X equals value |
| `${{ inputs.X != 'value' }}` | X not equals value |
| `${{ inputs.X == true }}` | Boolean is true |
| `${{ inputs.X.Y == 'value' }}` | Nested comparison |

## Next Steps

- [Architecture](ARCHITECTURE.md) - System design details
- [Local Development](LOCAL_DEVELOPMENT.md) - Running locally
