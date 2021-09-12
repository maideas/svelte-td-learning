<script>
  import PolicyModelShell from "./PolicyModelShell.svelte";

  const epsilon = 0.1; // exploration probability
  const alpha = 0.2; // learning rate
  const gamma = 0.9; // future reward discount factor

  export let numX;
  export let numY;
  export let numA;
  export let useNet = false; // we reused the Net as logits model

  let learningRate = 0.001;

  let stepTimer;
  let steps = 0;
  let rewardSum = 0;
  let done = false;

  let ModelShellComp;

  //====================================================
  // agent callback functions
  //====================================================

  export let envStepFunc = null; // (state, a) -> (nextState, r, done)
  export let episodeDoneFunc = null; // (steps, rewardSum) -> ()

  //====================================================
  // Policy Gradient algorithm
  //====================================================

  const shiftLogits = logits => {
    // for better numeric stability (http://cs231n.github.io/linear-classify/)
    let m = Math.max(...logits);
    let shiftedLogits = [];
    logits.forEach(e => {
      shiftedLogits.push(e - m);
    });
    return shiftedLogits;
  };

  const calcLogits = (logits, stepData) => {
    // calculate action probabilities from logits
    let actionProbs = ModelShellComp.softmaxForward(logits);

    // calculate action probability gradients
    // loss(a) = -log(a)*G  ->  grad(a) = loss'(a) = -G / a
    let actionGrads = Array(actionProbs.length).fill(0);
    actionGrads[stepData.a] = -stepData.g / (actionProbs[stepData.a] + 1e-6);

    // calculate logits gradients
    let logitsGrads = ModelShellComp.softmaxBackward(actionGrads);

    // apply logits gradients
    let adaptedLogits = [];
    for (let n = 0; n < logits.length; n++) {
      adaptedLogits.push(logits[n] - logitsGrads[n]);
    }

    // shift logits max to 0 for better stability ...
    return shiftLogits(adaptedLogits);
  };

  let trajectory;

  const PolicyGradientModelUpdate = async () => {
    for (let i = trajectory.length - 1; i >= 0; i--) {
      // calculate accumulated future reward for each trajectory step ...
      if (trajectory[i].done) {
        trajectory[i].g = trajectory[i].r;
      } else {
        trajectory[i].g = trajectory[i].r + gamma * trajectory[i + 1].g;
      }

      // update of the model, using the data calculated above ...
      let state = trajectory[i].state;
      let a = trajectory[i].a;
      let g = trajectory[i].g;

      ModelShellComp.updateModel({ state, a, g }, calcLogits);
    }

    // change to the updated model ...
    await ModelShellComp.takeModel();
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
        trajectory.push({ state, a, r, done });
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

  export const getValues = state => {
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
