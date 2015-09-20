OculomotorControl
=================

This repository holds the relevant code for the BrainScaleS "Demo2" WP6 Task2 Oculomotor control closed-loop


Before running the training procedure, a training set with visual stimuli needs to be generated.
This training set represents different moving stimuli which are going to be presented to the system one after another.
This is done by running:

    python create_training_stimuli.py

    Important parameters: 
        n_rf, n_v for the shape of the sensory layer
        n_training_x, n_training_v
        delay_input, delay_output


create_training_stimuli.py prints as last line:
Saving training stimuli parameters to: [TRAINING_PARAMETERS_FILE]

Then, run the training by running:
    mpirun -np [N] python main_training_reward_based_new.py [TRAINING_PARAMETERS_FILE]

    [N] is the number of processors

Testing:
    mpirun -np [N] python main_testing.py [TRAINING_FOLDER]
