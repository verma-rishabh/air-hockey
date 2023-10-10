from air_hockey_agent.TD3_agent import TD3_agent 


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

    agent = TD3_agent(env_info,agent_id=1)  
    # agent.load("hit/models/models/exp-v03-TD3-sanity_check_7dof-hit_0")
    return agent
