"""Configuration for speculative tool calling framework."""
import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class SpecConfig:
    """Configuration for speculative execution."""
    # Execution mode
    enable_speculation: bool = True  # Set to False for baseline actor-only mode

    # Models
    actor_model: str = "gpt-5"
    spec_model: str = "gpt-5-mini"
    model_provider: str = "openai"  # Kept for backward compatibility (mirrors actor provider)
    actor_provider: str = "openai"
    spec_provider: str = "openai"
    verbose_logging: bool = True
    actor_api_key: Optional[str] = None
    spec_api_key: Optional[str] = None
    embedding_api_key: Optional[str] = None

    # Speculation parameters
    top_k_spec: int = 3
    conf_threshold: float = 0.35

    # Execution limits
    max_steps: int = 12

    # LLM parameters
    llm_max_tokens: int = 1024

    # Dataset configuration
    dataset: str = "gaia"  # "gaia" or "hotpot"
    dataset_path: Optional[str] = None  # Custom path to dataset file
    dataset_size: int = 100  # Number of examples to load
    dataset_random_seed: int = 42  # Random seed for sampling

    # API Keys
    gensee_api_key: str = ""  # Gensee AI API key for web search

    @classmethod
    def from_env(cls) -> "SpecConfig":
        """Load configuration from environment variables."""
        base_provider = os.getenv("GAIA_MODEL_PROVIDER", "openai")
        actor_provider = os.getenv("GAIA_ACTOR_PROVIDER", base_provider)
        spec_provider = os.getenv("GAIA_SPEC_PROVIDER", base_provider)
        verbose_logging = os.getenv("GAIA_VERBOSE_LOGGING", "1") != "0"
        actor_api_key = os.getenv("GAIA_ACTOR_API_KEY")
        spec_api_key = os.getenv("GAIA_SPEC_API_KEY")
        embedding_api_key = os.getenv("GAIA_EMBEDDING_API_KEY")

        return cls(
            # Execution mode
            enable_speculation=(os.getenv("DISABLE_SPECULATION", "0") != "1"),

            # Models
            actor_model=os.getenv("GAIA_ACTOR_MODEL", "gpt-5"),
            spec_model=os.getenv("GAIA_SPEC_MODEL", "gpt-5-mini"),
            model_provider=actor_provider,
            actor_provider=actor_provider,
            spec_provider=spec_provider,
            verbose_logging=verbose_logging,
            actor_api_key=actor_api_key,
            spec_api_key=spec_api_key,
            embedding_api_key=embedding_api_key,

            # Speculation parameters
            top_k_spec=int(os.getenv("GAIA_TOPK", "3")),
            conf_threshold=float(os.getenv("GAIA_CONF_TH", "0.35")),

            # Execution limits
            max_steps=int(os.getenv("GAIA_MAX_STEPS", "12")),

            # Dataset configuration
            dataset=os.getenv("DATASET", "gaia").lower(),
            dataset_path=os.getenv("DATASET_PATH", None),
            dataset_size=int(os.getenv("DATASET_SIZE", "100")),
            dataset_random_seed=int(os.getenv("DATASET_RANDOM_SEED", "42")),

            # API Keys
            gensee_api_key=os.getenv("GENSEE_API_KEY", ""),
        )

    def is_gaia(self) -> bool:
        """Check if using GAIA dataset."""
        return self.dataset == "gaia"

    def is_hotpot(self) -> bool:
        """Check if using HotPotQA dataset."""
        return self.dataset == "hotpot"

    def print_config(self) -> None:
        """Print current configuration."""
        print("\n⚙️  Configuration:")
        print("="*60)
        print(f"  Speculation: {'ENABLED' if self.enable_speculation else 'DISABLED'}")
        print(f"  Actor Model:  {self.actor_model} ({self.actor_provider})")
        print(f"  Spec Model:   {self.spec_model} ({self.spec_provider})")
        if self.actor_provider == self.spec_provider:
            print(f"  Provider:     {self.actor_provider}")
        else:
            print(f"  Providers:    actor={self.actor_provider}, spec={self.spec_provider}")
        print(f"  Verbose Logs: {'ON' if self.verbose_logging else 'OFF'}")
        print(f"  Dataset:      {self.dataset}")
        print(f"  Dataset Size: {self.dataset_size}")
        print(f"  Max Steps:    {self.max_steps}")
        print("="*60)


# Global config instance
config = SpecConfig.from_env()
