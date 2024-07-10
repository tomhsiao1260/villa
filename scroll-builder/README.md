# scroll-builder
Scroll data transformation scripts.

Runs the server-side processing on the scroll data.
Updates the derived data if the input or the computation script changes.

## How to add a computation script

### Docker

The repository uses docker images for computation of derrived data.
Make sure to include a valid Dockerfile.

File access control is enforced with restricting the input directories to read-only when mounting to the docker image.

### Execution

The execution is determined by the builder.yaml file.
In it you have predifined regex helpers.
For example: scroll: [Scroll*/*.volpkg/]
With these you can build your execution flow for your computation.
The data list stores all expressions that will be matched to the server directories. All permutations of data_1 - data_n will be one call to the scipt each.
The script will then be called with the arguments specified under args.
The yaml definition looks like this

```yaml
scroll: [Scroll*/*.volpkg/]

...

scripts: 

    [name]:
    - location: [location]
    - [docker_image]
    - [script].py
    - data:
        - data_1: [regex_1/$(scroll)]
        - ...
        - data_n: [regex_n]
    - args:
        - volume: [xyz/$(data_i)/$(scroll)]
        - ...
    - read-only:
        - path_1
        - ...
        - path_m
    - read-write:
        - path_1
        - ...
        - path_k
```
