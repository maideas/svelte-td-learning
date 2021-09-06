<script>
  import PolicyModelShell from "./PolicyModelShell.svelte";

  const epsilon = 0.1; // exploration probability
  const alpha = 0.2; // learning rate
  const gamma = 1.0; // future reward discount factor

  export let numX;
  export let numY;
  export let numA;
  export let useNet = false; // we reused the Net as logits model

  let learningRate = 0.001;

  let stepTimer;
  let steps = 0;
  let rewardSum = 0;
  let done = false;

  let actionProbs = Array.from({ length: numX }, () =>
    Array.from({ length: numY }, () => null)
  );

  let ModelShellComp;

  //====================================================
  // agent callback functions
  //====================================================

  export let envStepFunc = null; // (state, a) -> (nextState, r, done)
  export let episodeDoneFunc = null; // (steps, rewardSum) -> ()

  //====================================================
  // Policy Gradient algorithm
  //====================================================

  const calcLogits = (logits, stepData) => {
    // calculate action probabilities from logits
    let actionProbs = ModelShellComp.softmaxForward(logits);

    // calculate action probability gradients
    let actionGrads = Array(actionProbs.length).fill(0);
    actionGrads[stepData.a] = -Math.log(actionProbs[stepData.a]) * stepData.g;

    // calculate logits gradients
    let logitsGrads = ModelShellComp.softmaxBackward(actionGrads);

    // apply logits gradients
    let newLogits = [];
    for (let n = 0; n < logits.length; n++) {
      newLogits.push(logits[n] + logitsGrads[n]);
    }

    if (useNet) {
      // pure newLogits are returned in case a Net is used, because the network
      // learning rate replaces the table variant alpha parameter ...
      return newLogits;
    }

    let tableLogits = [];
    for (let n = 0; n < logits.length; n++) {
      tableLogits.push((1.0 - alpha) * logits[n] + alpha * newLogits[n]);
    }
    return tableLogits;
  };

  let trajectory;

  const PolicyGradientModelUpdate = async () => {
    // calculate accumulated future reward for each trajectory step
    for (let n = 0; n < trajectory.length; n++) {
      let i = trajectory.length - 1 - n;

      if (n > 0) {
        trajectory[i].g = trajectory[i].r + gamma * trajectory[i + 1].g;
      } else {
        trajectory[i].g = trajectory[i].r;
      }
    }

    // calculate accumulated reward max and min values
    let g = trajectory[0].g;
    let g_max = g;
    let g_min = g;
    for (let n = 1; n < trajectory.length; n++) {
      let g = trajectory[n].g;
      if (g_max < g) g_max = g;
      if (g_min > g) g_min = g;
    }

    // calculate accumulated reward mean and distance
    let g_mean = (g_max + g_min) / 2.0;
    let g_dist = (g_max - g_min) / 2.0;
    if (g_dist < 1e-6) {
      g_dist = 1e-6;
    }

    // normalize accumulated reward values and update the policy model
    for (let n = 0; n < trajectory.length; n++) {
      let state = trajectory[n].state;
      let a = trajectory[n].a;
      let g = (trajectory[n].g - g_mean) / g_dist;

      await ModelShellComp.updateModel({ state, a, g }, calcLogits);
    }
  };

  const runEpisodeStep = state => {
    let stateNext;
    let a, r;

    if (done) {
      // policy-gradient-model is updated only after complete episode ...
      PolicyGradientModelUpdate();
      episodeDoneFunc(steps, rewardSum);
    } else {
      stepTimer = setTimeout(() => {
        a = ModelShellComp.getEpsilonGreedyAction(state, epsilon);
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
    ModelShellComp.init();
  };

  export const halt = () => {
    if (stepTimer) {
      clearTimeout(stepTimer);
    }
  };

  export const getQValues = state => {
    return ModelShellComp.getActionProbs(state);
  };
</script>

<PolicyModelShell
  {numX}
  {numY}
  {numA}
  {useNet}
  {learningRate}
  on:modelChanged
  bind:this={ModelShellComp} />
