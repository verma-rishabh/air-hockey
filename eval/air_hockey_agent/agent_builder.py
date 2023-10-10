from air_hockey_agent.TQC_agent_eval import TQC_agent 
from air_hockey_agent.TQC_agent_eval_3dof import TQC_agent_3dof 


def build_agent(env_info, **kwargs):
    """
    Function where an Agent that controls the environments should be returned.
    The Agent should inherit from the mushroom_rl Agent base env.

    Args:
        env_info (dict): The environment information
        kwargs (any): Additionally setting from agent_config.yml
    Returns:
         (AgentBase) An instance of the Agent
    """
    if "3dof-hit" in env_info["env_name"]:
        agent = TQC_agent_3dof(env_info,agent_id=1)
        agent.load("3dof_hit/models/tqc-hit-mushroomrl_3dof-hit_0")

    elif "3dof-defend" in env_info["env_name"]:
        agent = TQC_agent_3dof(env_info,agent_id=1)
        agent.load("3dof_defend/models/tqc-defend-mushroomrl_3dof-defend_0")

    if "7dof-hit" in env_info["env_name"]:
        agent = TQC_agent(env_info,agent_id=1)
        agent.load("7dof_hit/models/tqc-hit-mushroomrl_7dof-hit_0")

    if "7dof-defend" in env_info["env_name"]:
        agent = TQC_agent(env_info,agent_id=1)
        agent.load("7dof_defend/models/tqc-defend-mushroomrl_7dof-defend_0")

    if "7dof-prepare" in env_info["env_name"]:
        agent = TQC_agent(env_info,agent_id=1)
        agent.load("7dof_prepare/models/tqc-prepare-mushroomrl_7dof-prepare_0")
        
    return agent
