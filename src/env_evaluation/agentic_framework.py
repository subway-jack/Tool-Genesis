import os
import sys
from typing import Dict, Tuple

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.core.agents import BaseAgent, ChatAgent
from src.core.models import ModelFactory
from src.core.types import ModelPlatformType, ModelType
from src.core.toolkits import (
    FunctionTool,
    MCPServerToolsToolkit,
)

# Default solver/judge configuration (can be overridden via env vars)
_SOLVER_PLATFORM = os.environ.get("SOLVER_PLATFORM", "BAILIAN")
_SOLVER_MODEL = os.environ.get("SOLVER_MODEL", "BAILIAN_QWEN3_14B")
_JUDGE_PLATFORM = os.environ.get("JUDGE_PLATFORM", "")  # empty = same as solver
_JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "")


def initialize_agent(
    server_name: str,
    simulation_toolkit: MCPServerToolsToolkit,
    results_base_dir: str = "temp/agentic"
) -> Dict[str, BaseAgent]:

    ## initialize models
    solver_platform = getattr(ModelPlatformType, _SOLVER_PLATFORM, ModelPlatformType.BAILIAN)
    solver_model = getattr(ModelType, _SOLVER_MODEL, ModelType.BAILIAN_QWEN3_14B)
    simulate_solver_model = ModelFactory.create(
        model_platform=solver_platform,
        model_type=solver_model,
        model_config_dict={"temperature": 0},
    )

    judge_platform_name = _JUDGE_PLATFORM or _SOLVER_PLATFORM
    judge_model_name = _JUDGE_MODEL or _SOLVER_MODEL
    judge_platform = getattr(ModelPlatformType, judge_platform_name, ModelPlatformType.BAILIAN)
    judge_model_type = getattr(ModelType, judge_model_name, ModelType.BAILIAN_QWEN3_14B)
    judge_model = ModelFactory.create(
        model_platform=judge_platform,
        model_type=judge_model_type,
        model_config_dict={"temperature": 0},
    )
    
    ## initialize toolkits
    
    
    
    ## initialize agents
        
    simulate_solver_agent = ChatAgent(
        system_message="You are a helpful assistant.",
        model=simulate_solver_model,
        tools=simulation_toolkit.get_tools(),
        auto_save=True,
        results_base_dir=results_base_dir + "/simulate_solve/",
    )
    
    judge_agent = ChatAgent(
        system_message="You are a helpful assistant.",
        model=judge_model,
        tools=[],
        auto_save=True,
        results_base_dir=results_base_dir + "/judge/",
    )
    return {"simulate_solve": simulate_solver_agent, "judge": judge_agent}
