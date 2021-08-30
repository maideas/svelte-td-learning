<script>
  import { onMount } from "svelte";
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
  // Expected SARSA algorithm
  //====================================================

  const calcQValue = stepData => {
    let g;
    if (stepData.done) {
      g = stepData.r;
    } else {
      let vNextExpected = 0.0;
      let prob;
      // aNext is not used to calculate the expected next state value

      // each action has a base probability to be selected of epsilon divided
      // by the number of actions (random action selection case of e-greedy)
      prob = epsilon / numA;
      for (let a = 0; a < numA; a++) {
        vNextExpected +=
          prob * QModelShellComp.getQValue(stepData.stateNext, a);
      }

      // the maximum Q action has a probability of (1 - epsilon)
      // to be selected (greedy action selection case of e-greedy)
      prob = 1.0 - epsilon;
      let aNextGreedy = QModelShellComp.getPolicy(stepData.stateNext);
      vNextExpected +=
        prob * QModelShellComp.getQValue(stepData.stateNext, aNextGreedy);

      // G with respect to expected value V of next state
      g = stepData.r + gamma * vNextExpected;
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

  const runEpisodeStep = (state, a) => {
    let stateNext;
    let aNext, r;

    if (done) {
      episodeDoneFunc(steps, rewardSum);
    } else {
      stepTimer = setTimeout(async () => {
        [stateNext, r, done] = envStepFunc(state, a);
        aNext = QModelShellComp.getEpsilonGreedyAction(stateNext, epsilon);
        await QModelShellComp.updateModel(
          { state, a, r, stateNext, aNext, done },
          calcQValue
        );
        state = [...stateNext];
        a = Number(aNext);

        rewardSum += r;
        steps++;
        runEpisodeStep(state, a);
      }, 0);
    }
  };

  export const runEpisode = state => {
    rewardSum = 0;
    steps = 0;
    done = false;
    let a = QModelShellComp.getEpsilonGreedyAction(state, epsilon);
    runEpisodeStep(state, a);
  };

  export const init = () => {
    QModelShellComp.init();
  };

  onMount(() => {
    init();
  });

  export const halt = () => {
    if (stepTimer) {
      clearTimeout(stepTimer);
    }
  };

  export const getQValues = state => {
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
