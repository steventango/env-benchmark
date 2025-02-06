def get_env(env: str):
    if env == "jbw":
        from envs.jbw import JBWEnv
        return JBWEnv
    elif env == "forager":
        from envs.forager import Env as ForagerEnv
        return ForagerEnv
    else:
        raise ValueError(f"Unknown environment: {env}")
