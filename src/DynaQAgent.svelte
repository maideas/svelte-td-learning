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
  export let planningSteps = 10; // Dyna-Q parameter

  let learningRate = 0.002;
  let trainDistance = 1;

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
  // Dyna-Q algorithm
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

  let envModel = Array.from({ length: numX }, () =>
    Array.from({ length: numY }, () => Array.from({ length: numA }, () => null))
  );
  let seenStateActions = [];

  const DynaQModelUpdate = stepData => {
    let x = stepData.state[0];
    let y = stepData.state[1];
    let a = stepData.a;
    let seen = false;
    for (let n = 0; n < seenStateActions.length; n++) {
      if (
        seenStateActions[n][0] == x &&
        seenStateActions[n][1] == y &&
        seenStateActions[n][2] == a
      ) {
        seen = true;
        break;
      }
    }
    if (!seen) {
      seenStateActions.push([stepData.state, a]);
    }
    envModel[x][y][a] = [stepData.stateNext, stepData.r, stepData.done];
  };

  const DynaQGetModelStateAction = () => {
    let i = QModelShellComp.getRandomInt(seenStateActions.length);
    return seenStateActions[i];
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
        DynaQModelUpdate({ state, a, r, stateNext, done });
        state = [...stateNext];

        // planning steps (model based Q table update steps)
        for (let n = 0; n < planningSteps; n++) {
          let mState;
          let ma, mr;
          let mStateNext;
          let mDone;
          [mState, ma] = DynaQGetModelStateAction();
          [mStateNext, mr, mDone] = envModel[mState[0]][mState[1]][ma];
          await QModelShellComp.updateModel(
            { state: mState, a: ma, r: mr, stateNext: mStateNext, done: mDone },
            calcQValue
          );
        }

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
  {learningRate}
  on:modelChanged
  bind:this={QModelShellComp} />
