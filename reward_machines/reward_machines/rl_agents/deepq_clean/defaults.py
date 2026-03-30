def seaquest_environment():
    return dict(
        network='mlp',
        num_layers=3,
        num_hidden=512,
        lr=1e-4,
        buffer_size=100000,
        exploration_fraction=0.1,
        exploration_final_eps=0.01,
        train_freq=1,
        batch_size=32,
        learning_starts=5000,
        target_network_update_freq=1000,
        gamma=0.99,
        print_freq=1000,
    )
