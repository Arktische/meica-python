# MEICA Python Package

An all-in-one package makes AI training much easier.

- **trainer**: Config-driven training framework, you **only need to write the forward logic**.
- Coming soon:
  - Diffusion/FlowMatching trainer support
    - QwenImageEdit
    - ZImage
    - ...
  - non-PyTorch trainer support
    - ...
---

## Installation

```bash
pip install meica
```

Or from source:

```bash
git clone https://github.com/Arktische/meica-python.git
cd meica
pip install -e .
```
---

## Quick Start & Explained Example
If you prefer a quick start, run the example:
```python
python example.py --config config/trainer.yaml config/hypervar.yaml
```

### 1. Write two YAML config files

- Best Practice
  - split training config into two files: one for *training components*, one for *volatile hyperparameters*.
- Why?
  - enables ablation studies, variant switching, and fast tuning without touching the component topology.
- Components for training:
  - model
  - dataset/dataloader
  - optimizer
  - LR scheduler
  - ...
- Hyperparameters
  - `epoch`, `batch_size`, `lr`, `max_grad_norm`, etc., and can be reused in the components config via `${...}` references.

#### [hypervar.yaml](config/hypervar.yaml)
```yaml
epoch: 2
max_grad_norm: 1.0
batch_size: 2
lr: 0.1
```


#### [trainer.yaml](config/trainer.yaml)

```yaml
checkpoint_every_n_steps: 1
checkpoint_dir: "checkpoints"

module:
  type: torch.nn.Linear
  args:
    in_features: 2
    out_features: 1
  requires_grad_: true

train_dataloader:
  type: torch.utils.data.DataLoader
  args:
    dataset:
      type: example.MyDataset
      args:
        num: 4
    # value reference from hypervar.yaml
    batch_size: ${batch_size}
    shuffle: true
    num_workers: 1
    drop_last: false

optimizer:
  type: torch.optim.SGD
  args:
    lr: ${lr}               
    params:
      # equivalent to:
      # call module.parameters()
      object: ${module} 
      parameters:

lr_scheduler:
  type: torch.optim.lr_scheduler.StepLR
  args:
    optimizer: ${optimizer} # object reference
    step_size: 1
```


### 2. Implement a custom Trainer
```python
import torch.nn.functional as F
from meica import Trainer
class MyTrainer(Trainer):
    def train_step(self, batch, step):
        x, y = batch
        out = self.module(x)
        loss = F.mse_loss(out, y)
        return loss

trainer = MyTrainer()
trainer.configure("config/trainer.yaml", "config/hypervar.yaml")
trainer.fit()
```

---

## YAML Configuration Syntax

### Construct objects

Use `type` + `args` to build objects:

```yaml
module:
  type: torch.nn.Linear
  args:
    in_features: 2
    out_features: 1
```

### Call methods / set attributes

Extra keys become method calls or attribute assignments:

```yaml
module:
  type: torch.nn.Linear
  args: { in_features: 10, out_features: 10 }
  requires_grad_: false

weights:
  type: torch.load
  args:
    - "weights.pth"

module_with_weights:
  type: torch.nn.Linear
  args: { in_features: 10, out_features: 10 }
  load_state_dict:
    state_dict: ${weights}
    strict: true
```

### Reuse existing objects

Wrap an existing instance with `object` and call its methods:

```yaml
optimizer:
  type: torch.optim.SGD
  args:
    lr: 0.1
    params:
      object: ${module}
      trainable_parameters:
```

### Reference syntax

- Full reference: exactly `${path.to.node}` uses the target value as-is

```yaml
lr_scheduler:
  type: torch.optim.lr_scheduler.StepLR
  args:
    optimizer: ${optimizer}
    step_size: 1
```

- String interpolation: `${...}` inside a larger string inserts the string form

```yaml
name:
  type: builtins.str
  args: ["demo"]

log_dir: "runs/${name}"
```

- Escape literal `${...}`: use `$${`

```yaml
text: "$${value}"            # becomes "${value}"
mixed: "show $${x} and ${x}" # escapes and interpolates together
```

- Paths and list indices are supported

```yaml
values: [10, 20]
second: "${values[1]}"

# Nested path + list index
a:
  b:
    - 1
    - 2
c: "${a.b[0]}"   # -> 1

# Deeper chain with object fields in list items
d:
  e:
    - { foo: "bar" }
    - { foo: "baz" }
first_foo: "${d.e[0].foo}"  # -> "bar"
```
---
