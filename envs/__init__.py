def get_env(env: str):
    if env == "jbw":
        from envs.jbw import JBWEnv
        return JBWEnv
    elif env == "forager":
        from envs.forager import Env as ForagerEnv
        return ForagerEnv
    elif env == "foragax":
        from envs.foragax import ForagaxEnv
        return ForagaxEnv
    else:
        raise ValueError(f"Unknown environment: {env}")
