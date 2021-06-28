
def get_model(sess, hps):
    if hps.model == 'acflow':
        from .acflow import Model
        model = Model(sess, hps)
    elif hps.model == 'acflow_classifier':
        from .acflow_classifier import Model
        model = Model(sess, hps)
    elif hps.model == 'acflow_regressor':
        from .acflow_regressor import Model
        model = Model(sess, hps)
    elif hps.model == 'acnp_classifier':
        from .acnp_classifier import Model
        model = Model(sess, hps)
    elif hps.model == 'ac_reg':
        from .ac_reg import Model
        model = Model(sess, hps)
    elif hps.model == 'ac_cls':
        from .ac_cls import Model
        model = Model(sess, hps)
    elif hps.model == 'ac_cls_env':
        from .ac_cls_env import Model
        model = Model(sess, hps)
    elif hps.model == 'ac_reg_env':
        from .ac_reg_env import Model
        model = Model(sess, hps)
    elif hps.model == 'ac_dag_env':
        from .ac_dag_env import Model
        model = Model(sess, hps)
    else:
        raise Exception()

    return model