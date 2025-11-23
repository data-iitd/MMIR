# Configuration

## Aim
This directory holds the configuration files for the benchmark. It defines paths, models, endpoints, and other parameters used across the project.

## Files

### `config.yaml`
The main configuration file. Key sections include:
-   `paths`: Defines dataset paths (COCO, Flickr) and the project root.
-   `models`: Specifies model paths and variants (CLIP, MiniLM, FLAVA, UniIR, BLIP2, FLMR).
-   `selected_config`: Default values for dataset, model, etc.
-   `vector_store`: Configuration for Solr, FAISS, and FLMR vector stores.
-   `endpoints`: URLs for various retrieval services and endpoints.

### `config_utils.py`
A utility module to load and update the configuration.
-   `load_config(path)`: Loads the YAML config.
-   `save_config(config, path)`: Saves the config to YAML.
-   `update_config_section(section, key, value, path)`: Updates a specific section in the config.
