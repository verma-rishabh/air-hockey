## Installation

[ [Website] ](https://air-hockey-challenge.robot-learning.net/) [ [Documentation] ](https://air-hockey-challenges-docs.readthedocs.io/en/latest/)  [ [Discussion] ](https://github.com/AirHockeyChallenge/air_hockey_challenge/discussions)

This is the source code for the Air Hockey Challenge.

The current released version is **Qualifying**.

Please see the [documentation](https://air-hockey-challenges-docs.readthedocs.io/en/latest/) for Installation and usage.



## Evaluation
``python run.py -r -e 7dof-defend --n_cores 1 --n_episodes 100 --example tqc_mpc``

``python run.py -r -e 7dof-defend --n_cores 1 --n_episodes 100 --example tqc_ik``

<!-- ![TQC-IK][tqc_ik.gif]

![TQC-MPC][tqc_mpc.gif] -->

### TQC + MPC
<p align="left">
  <img src="tqc_mpc.gif" height="256">

</p>


### TQC + IK
<p align="left">
  <img src="tqc_ik.gif" height="256">

</p>

## License

[MIT License](https://github.com/AirHockeyChallenge/air_hockey_challenge/blob/warm-up/LICENSE)