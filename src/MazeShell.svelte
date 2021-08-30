<script>
  import { numA } from "./MazeTile.svelte";
  import Maze from "./Maze.svelte";
  import LinePlot from "./LinePlot.svelte";

  import QLearningAgent from "./QLearningAgent.svelte";
  import SarsaAgent from "./SarsaAgent.svelte";
  import ExpectedSarsaAgent from "./ExpectedSarsaAgent.svelte";
  import DynaQAgent from "./DynaQAgent.svelte";
  import MonteCarloAgent from "./MonteCarloAgent.svelte";

  //====================================================

  export let blocked = [];
  export let terminal = Array([0, 0]);
  export let rewards = Array([0, 0, 1.0]); // [x, y, reward]
  export let defaultReward = 0;
  export let startState = undefined;
  export let numEpisodes = 1000;
  export let planningSteps = 10; // Dyna-Q parameter

  export let numX = 5;
  export let numY = 5;

  //====================================================

  let agentComp;
  let mazeComp;
  let plotComp;

  let stepsPerEpisode = [];
  let rewardPerEpisode = [];
  let selectedAlgorithm;

  let episodeTimer;
  let episode = 0;

  //====================================================

  let useQNet = false;
  let duelingQNet = false;
  let runDisabled = false;
  let useQNetDisabled = false;
  let duelingQNetDisabled = true;

  //====================================================
  // agent callback functions
  //====================================================

  const envStepFunc = (state, a) => {
    return mazeComp.step(state, a);
  };

  const modelChangedFunc = () => {
    for (let y = 0; y < numY; y++) {
      for (let x = 0; x < numX; x++) {
        let state = [x, y];
        let QValues = agentComp.getQValues(state);
        mazeComp.setQValues(state, QValues);
      }
    }
  };

  const episodeDoneFunc = (steps, rewardSum) => {
    stepsPerEpisode.push(steps);
    rewardPerEpisode.push(rewardSum);
    plotComp.updatePlot();
    runEpisode();
  };

  //====================================================

  const algorithms = [
    { name: "Q-Learning" },
    { name: "SARSA" },
    { name: "Expected SARSA" },
    { name: "Dyna-Q" },
    { name: "Monte Carlo" }
  ];

  const runEpisode = () => {
    runDisabled = true;
    episodeTimer = setTimeout(() => {
      if (episode < numEpisodes) {
        episode++;
        let state = mazeComp.getRandomStartState();
        if (startState) {
          state = startState;
        }
        agentComp.runEpisode(state);
      } else {
        runDisabled = false;
      }
    }, 0);
  };

  const halt = () => {
    if (episodeTimer) {
      clearTimeout(episodeTimer);
    }
    agentComp.halt();
    runDisabled = false;
  };

  const init = () => {
    duelingQNetDisabled = !useQNet;
    if (duelingQNetDisabled) {
      duelingQNet = false;
    }
    episode = 0;
    stepsPerEpisode = [];
    rewardPerEpisode = [];
    plotComp.clearPlot();
    agentComp.init();
  };
</script>

<style>
  div.container {
    -webkit-box-sizing: border-box;
    -moz-box-sizing: border-box;
    box-sizing: border-box;

    margin: 30px auto;
    padding: 5px;
    border: 3px solid #eee;
  }
  div.box {
    display: flex;
    justify-content: center;
    margin: 20px;
  }
  select {
    margin: 0px 5px;
    color: inherit;
    background: inherit;
  }
  div.narrow-box {
    margin: 0px;
  }
  button {
    margin: 0px 5px;
  }
  input {
    margin: 5px;
  }
  div.flexrow {
    display: flex;
    flex-direction: row;
    align-items: center;
    margin: 0 5px;
  }
  div.flexrow.disabled {
    opacity: 0.6;
  }
</style>

<div class="container" style="width: {16 + numX * 100 + (numX - 1) * 4}px;">
  <div class="narrow-box">
    <LinePlot
      bind:this={plotComp}
      bind:data={rewardPerEpisode}
      bind:dataSecond={stepsPerEpisode}
      yTitle={'reward per episode'}
      ySecondTitle={'steps per episode'}
      hasSecondY={false} />
  </div>

  <div class="box">EPISODE : {episode}</div>
  <div class="box">
    <select bind:value={selectedAlgorithm} on:change={init}>
      {#each algorithms as algo}
        <option value={algo.name}>{algo.name}</option>
      {/each}
    </select>
    <div class="flexrow" class:disabled={useQNetDisabled}>
      <input
        type="checkbox"
        disabled={useQNetDisabled}
        bind:checked={useQNet}
        on:change={init} />
      DQN
    </div>
    <div class="flexrow" class:disabled={duelingQNetDisabled}>
      <input
        type="checkbox"
        disabled={duelingQNetDisabled}
        bind:checked={duelingQNet}
        on:change={init} />
      Dueling
    </div>
    <button on:click={init}>init</button>
    <button on:click={halt}>halt</button>
    <button on:click={runEpisode} disabled={runDisabled}>run</button>
  </div>

  <div class="narrow-box">
    <Maze
      {numX}
      {numY}
      {blocked}
      {terminal}
      {rewards}
      {defaultReward}
      bind:this={mazeComp} />
  </div>
</div>

{#if selectedAlgorithm == 'Q-Learning'}
  <QLearningAgent
    {numX}
    {numY}
    {numA}
    {useQNet}
    {duelingQNet}
    {envStepFunc}
    {episodeDoneFunc}
    on:modelChanged={modelChangedFunc}
    bind:this={agentComp} />
{:else if selectedAlgorithm == 'SARSA'}
  <SarsaAgent
    {numX}
    {numY}
    {numA}
    {useQNet}
    {duelingQNet}
    {envStepFunc}
    {episodeDoneFunc}
    on:modelChanged={modelChangedFunc}
    bind:this={agentComp} />
{:else if selectedAlgorithm == 'Expected SARSA'}
  <ExpectedSarsaAgent
    {numX}
    {numY}
    {numA}
    {useQNet}
    {duelingQNet}
    {envStepFunc}
    {episodeDoneFunc}
    on:modelChanged={modelChangedFunc}
    bind:this={agentComp} />
{:else if selectedAlgorithm == 'Dyna-Q'}
  <DynaQAgent
    {numX}
    {numY}
    {numA}
    {useQNet}
    {duelingQNet}
    {planningSteps}
    {envStepFunc}
    {episodeDoneFunc}
    on:modelChanged={modelChangedFunc}
    bind:this={agentComp} />
{:else if selectedAlgorithm == 'Monte Carlo'}
  <MonteCarloAgent
    {numX}
    {numY}
    {numA}
    {useQNet}
    {duelingQNet}
    {envStepFunc}
    {episodeDoneFunc}
    on:modelChanged={modelChangedFunc}
    bind:this={agentComp} />
{/if}
