<script>
  import LogitsTable from "./LogitsTable.svelte";
  import LogitsNet from "./LogitsNet.svelte";

  export let numX;
  export let numY;
  export let numA;
  export let useNet = true;
  export let learningRate = 0.002; // usually not needed to be overidden

  //====================================================
  // softmax
  //====================================================

  let softmax_y = [];

  export const softmaxForward = x => {
    // for better numeric stability (http://cs231n.github.io/linear-classify/)
    // -> max a value will be adjusted to 0 -> max e^a will be 1
    // This adaption does not change the result of the softmax calculation.
    let max_x = Math.max(...x);
    let a = [];
    x.forEach(e => {
      a.push(e - max_x);
    });
    let exp_a = [];
    a.forEach(e => {
      exp_a.push(Math.exp(e));
    });
    let sum = 0;
    exp_a.forEach(e => {
      sum += e;
    });
    softmax_y = [];
    exp_a.forEach(e => {
      softmax_y.push(e / sum);
    });
    return softmax_y;
  };

  export const softmaxBackward = grad_y => {
    // activation function derivative, used to pass gradients backward
    // n : softmax layer output index
    // k : softmax layer input index
    let grad_x = Array(grad_y.length).fill(0);
    for (let n = 0; n < grad_y.length; n++) {
      for (let k = 0; k < grad_y.length; k++) {
        if (n == k) {
          grad_x[k] += softmax_y[n] * (1.0 - softmax_y[k]) * grad_y[n];
        } else {
          grad_x[k] += softmax_y[n] * (0.0 - softmax_y[k]) * grad_y[n];
        }
      }
    }
    return grad_x;
  };

  //====================================================

  let LogitsModelComp;

  export const getActionProbs = state => {
    let logits = LogitsModelComp.predictState(state);
    return softmaxForward(logits);
  };

  export const getPolicy = state => {
    let actionProbs = getActionProbs(state);
    let rand = Math.random();
    let prob = 0;

    // sample an action from its probability distribution
    for (let a = 0; a < actionProbs.length; a++) {
      prob += actionProbs[a];
      if (rand < prob) {
        return a;
      }
    }
    return actionProbs.length - 1;
  };

  export const getRandomInt = n => {
    // result range [0 .. n-1]
    return Math.floor(Math.random() * Math.floor(n));
  };

  export const getEpsilonGreedyAction = (state, epsilon) => {
    if (Math.random() < epsilon) {
      return getRandomInt(numA); // choose random action with epsilon probability
    } else {
      return getPolicy(state); // else choose action according to current policy
    }
  };

  export const init = () => {
    LogitsModelComp.init();
  };

  export const updateModel = (stepData, calcLogitsFunc) => {
    return LogitsModelComp.updateModel(stepData, calcLogitsFunc);
  };

  export const takeModel = () => {
    return LogitsModelComp.takeModel();
  };
</script>

{#if useNet}
  <LogitsNet
    {numX}
    {numY}
    {numA}
    {learningRate}
    on:modelChanged
    bind:this={LogitsModelComp} />
{:else}
  <LogitsTable
    {numX}
    {numY}
    {numA}
    on:modelChanged
    bind:this={LogitsModelComp} />
{/if}
