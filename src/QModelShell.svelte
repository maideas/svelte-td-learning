<script>
  import QTable from "./QTable.svelte";
  import QNet from "./QNet.svelte";

  export let numX;
  export let numY;
  export let numA;
  export let useQNet = true;
  export let duelingQNet = false;
  export let learningRate = 0.002; // usually not needed to be overidden
  export let trainDistance = 10;

  let QModelComp;

  export const getQValues = state => {
    return QModelComp.predictState(state);
  };

  export const getQValue = (state, a) => {
    return getQValues(state)[a];
  };

  export const getMaxQValue = state => {
    return Math.max(...getQValues(state));
  };

  export const getPolicy = state => {
    let QValues = getQValues(state);
    return QValues.indexOf(Math.max(...QValues));
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
    QModelComp.init();
  };

  export const updateModel = async (stepData, calcQValueFunc) => {
    return QModelComp.updateModel(stepData, calcQValueFunc);
  };
</script>

{#if useQNet}
  <QNet
    {numX}
    {numY}
    {numA}
    {duelingQNet}
    {learningRate}
    {trainDistance}
    on:modelChanged
    bind:this={QModelComp} />
{:else}
  <QTable {numX} {numY} {numA} on:modelChanged bind:this={QModelComp} />
{/if}
