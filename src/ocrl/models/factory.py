from ocrl.models.world_model import RandomWorldStepGru

def CreateWorlModelInstance(config):
    world_model = RandomWorldStepGru(batch_size=config['world_model_training_params']['batch_size'],
                                    state_dim=config['robot_params']['state_dims'],
                                    contact_dim=config['robot_params']['contact_dims'],
                                    action_dim=config['robot_params']['action_dims'],
                                    embed_dim=config['world_model_arch_params']['embed_dim'],
                                    hidden_dim=config['world_model_arch_params']['gru_hidden_dim'],
                                    num_gru_layers=config['world_model_arch_params']['num_gru_layers'],
                                    mlp_dim=config['world_model_arch_params']['mlp_head_dim'],
                                    lr=config['world_model_training_params']['learning_rate'],
                                    weight_decay=config['world_model_training_params']['weight_decay'],
                                    device=config['device'])

    return world_model
