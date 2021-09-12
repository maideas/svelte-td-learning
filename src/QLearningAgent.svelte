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
  // Q-Learning algorithm
  //====================================================

  const calcQValue = stepData => {
    let g;
    if (stepData.done) {
      g = stepData.r;
    } else {
      // G with respect to maximum Q value (over all actions) of next state
      g = stepData.r + gamma * QModelShellComp.getMaxQValue(stepData.stateNext);
    }
    if (useQNet) {
      // pure g is returned in case a DQN is used, because the network
      // learning rate replaces the table variant alpha parameter ...
      return g;
    }
    return (
      (1.0 - alpha) * QModelShellComp.getQValue(stepData.state, stepData.a) +
      alpha * g
    );
  };

  const runEpisodeStep = state => {
    let stateNext;
    let a, r;

    if (done) {
      episodeDoneFunc(steps, rewardSum);
    } else {
      stepTimer = setTimeout(async () => {
        a = QModelShellComp.getEpsilonGreedyAction(state, epsilon);
        [stateNext, r, done] = envStepFunc(state, a);
        await QModelShellComp.updateModel(
          { state, a, r, stateNext, done },
          calcQValue
        );
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
  on:modelChanged
  bind:this={QModelShellComp} />
