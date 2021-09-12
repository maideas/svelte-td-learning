<script>
  import QModelShell from "./QModelShell.svelte";

  const epsilon = 0.1; // exploration probability
  const alpha = 0.2; // learning rate
  const gamma = 0.9; // future reward discount factor

  export let numX;
  export let numY;
  export let numA;
  export let useQNet = false;
  export let duelingQNet = false;

  let learningRate = 0.001;

  let stepTimer;
  let steps = 0;
  let rewardSum = 0;
  let done = false;

  let QModelShellComp;

  //====================================================
  // agent callback functions
  //====================================================

  export let envStepFunc = null; // (state, a) -> (nextState, r, done)
  export let episodeDoneFunc = null; // (steps, rewardSum) -> ()

  //====================================================
  // Monte Carlo algorithm
  //====================================================

  const calcQValue = stepData => {
    if (useQNet) {
      // pure g is returned in case a DQN is used, because the network
      // learning rate replaces the table variant alpha parameter ...
      return stepData.g;
    }
    return (
      (1.0 - alpha) * QModelShellComp.getQValue(stepData.state, stepData.a) +
      alpha * stepData.g
    );
  };

  let trajectory;

  const MonteCarloQModelUpdate = async () => {
    for (let y = 0; y < numY; y++) {
      for (let x = 0; x < numX; x++) {
        for (let a = 0; a < numA; a++) {
          let take = false;
          let g = 0.0;
          let gammaProduct = 1.0;

          for (let n = 0; n < trajectory.length; n++) {
            if (
              x == trajectory[n].state[0] &&
              y == trajectory[n].state[1] &&
              a == trajectory[n].a
            ) {
              take = true;
            }
            if (take) {
              g += gammaProduct * trajectory[n].r;
              gammaProduct *= gamma;
            }
          }

          if (take) {
            let state = [x, y];
            await QModelShellComp.updateModel({ state, a, g }, calcQValue);
          }
        }
      }
    }
  };

  const runEpisodeStep = state => {
    let stateNext;
    let a, r;

    if (done) {
      // Q-model is updated only after complete episode ...
      MonteCarloQModelUpdate();
      episodeDoneFunc(steps, rewardSum);
    } else {
      stepTimer = setTimeout(() => {
        a = QModelShellComp.getEpsilonGreedyAction(state, epsilon);
        [stateNext, r, done] = envStepFunc(state, a);
        trajectory.push({ state, a, r });
        state = [...stateNext];

        rewardSum += r;
        steps++;
        runEpisodeStep(state);
      }, 0);
    }
  };

  export const runEpisode = state => {
    rewardSum = 0;
    steps = 0;
    done = false;
    trajectory = [];
    runEpisodeStep(state);
  };

  export const init = () => {
    QModelShellComp.init();
  };

  export const halt = () => {
    if (stepTimer) {
      clearTimeout(stepTimer);
    }
  };

  export const getValues = state => {
    return QModelShellComp.getQValues(state);
  };
</script>

<QModelShell
  {numX}
  {numY}
  {numA}
  {useQNet}
  {duelingQNet}
  {learningRate}
  on:modelChanged
  bind:this={QModelShellComp} />
