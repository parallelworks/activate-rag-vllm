#!/usr/bin/env python3
"""
lib/config_validator.py - Configuration validation for ACTIVATE RAG-vLLM

Validates configuration before service launch to catch common issues early.

Usage:
    python3 lib/config_validator.py [--config env.sh] [--strict]
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Optional


class ConfigValidator:
    """Validates ACTIVATE RAG-vLLM configuration."""
    
    def __init__(self, strict: bool = False):
        self.strict = strict
        self.errors: list[str] = []
        self.warnings: list[str] = []
    
    def error(self, message: str) -> None:
        """Record an error."""
        self.errors.append(message)
    
    def warn(self, message: str) -> None:
        """Record a warning."""
        self.warnings.append(message)
        if self.strict:
            self.errors.append(f"(strict) {message}")
    
    def validate_all(self, config: dict) -> bool:
        """Run all validations."""
        self.validate_model_config(config)
        self.validate_port_config(config)
        self.validate_paths(config)
        self.validate_runtime(config)
        
        return len(self.errors) == 0
    
    def validate_model_config(self, config: dict) -> None:
        """Validate model configuration."""
        model_path = config.get("MODEL_PATH") or config.get("MODEL_NAME")
        model_source = config.get("MODEL_SOURCE", "local")
        
        if not model_path:
            self.error("MODEL_PATH or MODEL_NAME not specified")
            return
        
        if model_source == "local":
            path = Path(model_path)
            
            if not path.exists():
                self.error(f"Model path does not exist: {model_path}")
                return
            
            if not path.is_dir():
                self.error(f"Model path is not a directory: {model_path}")
                return
            
            # Check for required files
            config_file = path / "config.json"
            if not config_file.exists():
                self.error(f"Missing config.json in model directory: {model_path}")
            else:
                # Validate config.json is valid JSON
                try:
                    with open(config_file) as f:
                        model_config = json.load(f)
                    
                    # Check for model type
                    if "model_type" not in model_config:
                        self.warn("config.json missing 'model_type' field")
                    
                except json.JSONDecodeError as e:
                    self.error(f"Invalid JSON in config.json: {e}")
            
            # Check for tokenizer
            tokenizer_files = ["tokenizer.json", "tokenizer_config.json", "tokenizer.model"]
            has_tokenizer = any((path / f).exists() for f in tokenizer_files)
            if not has_tokenizer:
                self.warn(f"No tokenizer file found in {model_path}")
            
            # Check for weights
            weight_patterns = ["*.safetensors", "*.bin", "*.pt"]
            has_weights = any(list(path.glob(p)) for p in weight_patterns)
            if not has_weights:
                self.error(f"No model weight files found in {model_path}")
        
        elif model_source == "huggingface":
            # Validate HuggingFace model ID format
            if "/" not in model_path:
                self.warn(f"HuggingFace model ID should be in 'owner/model' format: {model_path}")
    
    def validate_port_config(self, config: dict) -> None:
        """Validate port configuration."""
        ports = {
            "VLLM_SERVER_PORT": config.get("VLLM_SERVER_PORT", "8000"),
            "PROXY_PORT": config.get("PROXY_PORT", "8081"),
            "RAG_PORT": config.get("RAG_PORT", "8080"),
            "CHROMA_PORT": config.get("CHROMA_PORT", "8001"),
        }
        
        # Convert to integers
        port_values = {}
        for name, value in ports.items():
            try:
                port_values[name] = int(value)
            except (ValueError, TypeError):
                self.error(f"Invalid port value for {name}: {value}")
                continue
        
        # Check for valid port range
        for name, port in port_values.items():
            if port < 1 or port > 65535:
                self.error(f"Port out of range for {name}: {port}")
            elif port < 1024:
                self.warn(f"Port {name}={port} requires root privileges")
        
        # Check for conflicts
        values = list(port_values.values())
        if len(values) != len(set(values)):
            self.error("Port conflict detected - duplicate port assignments")
    
    def validate_paths(self, config: dict) -> None:
        """Validate path configurations."""
        # DOCS_DIR
        docs_dir = config.get("DOCS_DIR")
        if docs_dir and docs_dir != "undefined":
            path = Path(docs_dir)
            if not path.exists():
                self.warn(f"DOCS_DIR does not exist (will be created): {docs_dir}")
        
        # HF_CACHE
        hf_cache = config.get("HF_CACHE") or config.get("HF_HOME")
        if hf_cache:
            path = Path(hf_cache)
            if not path.exists():
                self.warn(f"HuggingFace cache directory does not exist: {hf_cache}")
    
    def validate_runtime(self, config: dict) -> None:
        """Validate runtime configuration."""
        runmode = config.get("RUNMODE", "singularity")
        runtype = config.get("RUNTYPE", "all")
        
        if runmode not in ("docker", "singularity"):
            self.error(f"Invalid RUNMODE: {runmode} (expected: docker, singularity)")
        
        if runtype not in ("all", "vllm"):
            self.error(f"Invalid RUNTYPE: {runtype} (expected: all, vllm)")
        
        # vLLM extra args validation
        vllm_args = config.get("VLLM_EXTRA_ARGS", "")
        if vllm_args:
            # Check for common issues
            if "--model" in vllm_args:
                self.warn("VLLM_EXTRA_ARGS contains --model; use MODEL_NAME instead")
            if "--port" in vllm_args:
                self.warn("VLLM_EXTRA_ARGS contains --port; use VLLM_SERVER_PORT instead")
    
    def report(self) -> None:
        """Print validation report."""
        if self.errors:
            print("\n❌ Configuration validation FAILED:", file=sys.stderr)
            for error in self.errors:
                print(f"  ✗ {error}", file=sys.stderr)
        
        if self.warnings:
            print("\n⚠️  Warnings:", file=sys.stderr)
            for warning in self.warnings:
                print(f"  ⚡ {warning}", file=sys.stderr)
        
        if not self.errors and not self.warnings:
            print("✓ Configuration validation passed")
        elif not self.errors:
            print("\n✓ Configuration validation passed with warnings")


def load_config_from_env() -> dict:
    """Load configuration from environment variables."""
    return dict(os.environ)


def load_config_from_file(filepath: str) -> dict:
    """Load configuration from env.sh or .env file."""
    config = {}
    path = Path(filepath)
    
    if not path.exists():
        return config
    
    content = path.read_text()
    
    for line in content.splitlines():
        line = line.strip()
        
        # Skip comments and empty lines
        if not line or line.startswith("#"):
            continue
        
        # Remove 'export ' prefix if present
        if line.startswith("export "):
            line = line[7:]
        
        # Parse key=value
        if "=" in line:
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip()
            
            # Remove quotes
            if (value.startswith('"') and value.endswith('"')) or \
               (value.startswith("'") and value.endswith("'")):
                value = value[1:-1]
            
            config[key] = value
    
    return config


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Validate ACTIVATE RAG-vLLM configuration"
    )
    parser.add_argument(
        "--config", "-c",
        help="Path to configuration file (env.sh or .env)",
        default=None
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat warnings as errors"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config_from_env()
    
    if args.config:
        file_config = load_config_from_file(args.config)
        config.update(file_config)
    else:
        # Try default config files
        for default_file in ["env.sh", ".env", ".run.env"]:
            if Path(default_file).exists():
                file_config = load_config_from_file(default_file)
                config.update(file_config)
                break
    
    # Run validation
    validator = ConfigValidator(strict=args.strict)
    success = validator.validate_all(config)
    validator.report()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
