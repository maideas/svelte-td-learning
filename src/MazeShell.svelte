<script>
  import { onMount } from "svelte";
  import Maze from "./Maze.svelte";
  import LinePlot from "./LinePlot.svelte";

  const epsilon = 0.1; // exploration probability
  const alpha = 0.2; // learning rate
  const gamma = 0.9; // future reward discount factor

  const numA = 4; // number of possible actions (= number of directions)

  /*
  let blocked = Array([0, 0], [3, 2], [4, 1], [3, 3]);
  let terminal = Array([1, 1]);
  let rewards = Array([1, 1, 1]); // [x, y, reward]
  let defaultReward = -0.1;

  let numX = 6;
  let numY = 4;
*/

  //====================================================

  export let blocked = Array();
  export let terminal = Array([0, 0]);
  export let rewards = Array([0, 0, 1.0]); // [x, y, reward]
  export let defaultReward = 0;
  export let startState = undefined;
  export let numEpisodes = 500;
  export let planningSteps = 10; // Dyna-Q parameter

  export let numX = 5;
  export let numY = 5;

  //====================================================

  let timer;
  let mazeComp;
  let plotComp;
  let episode = 0;
  let stepsPerEpisode = [];
  let rewardPerEpisode = [];
  let algorithm;

  //====================================================

  const getRandomInt = n => {
    // result range [0 .. n-1]
    return Math.floor(Math.random() * Math.floor(n));
  };

  //====================================================
  // Q-Learning algorithm
  //====================================================

  const QLearningQTableUpdate = (x, y, a, r, xNext, yNext) => {
    let g;
    if (mazeComp.isTerminal(xNext, yNext)) {
      g = r;
    } else {
      // G with respect to maximum Q value (over all actions) of next state
      g = r + gamma * mazeComp.getMaxQValue(xNext, yNext);
    }
    let q = (1.0 - alpha) * mazeComp.getQValue(x, y, a) + alpha * g;
    mazeComp.setQValue(x, y, a, q);
  };

  const runQLearningEpisode = () => {
    let x, y;
    let xNext, yNext;
    let a, r;
    let max_steps = 100000;
    let steps = 0;
    let rewardSum = 0;

    if (startState) {
      [x, y] = startState;
    } else {
      [x, y] = mazeComp.getRandomStartState();
    }

    // run episode until terminal state has been reached
    while (!mazeComp.isTerminal(x, y)) {
      a = mazeComp.getEpsilonGreedyAction(x, y, epsilon);
      [xNext, yNext, r] = mazeComp.step(x, y, a);
      QLearningQTableUpdate(x, y, a, r, xNext, yNext);
      x = Number(xNext);
      y = Number(yNext);
      rewardSum += r;

      steps++;
      if (steps >= max_steps) {
        console.log("WARNING: Maximum number of steps per episode reached.");
        stepsPerEpisode.push(steps);
        rewardPerEpisode.push(rewardSum);
        return;
      }
    }
    stepsPerEpisode.push(steps);
    rewardPerEpisode.push(rewardSum);
  };

  //====================================================
  // SARSA algorithm
  //====================================================

  const SarsaQTableUpdate = (x, y, a, r, xNext, yNext, aNext) => {
    let g;
    if (mazeComp.isTerminal(xNext, yNext)) {
      g = r;
    } else {
      // G with respect to policy related Q value of next state
      g = r + gamma * mazeComp.getQValue(xNext, yNext, aNext);
    }
    let q = (1.0 - alpha) * mazeComp.getQValue(x, y, a) + alpha * g;
    mazeComp.setQValue(x, y, a, q);
  };

  const runSarsaEpisode = () => {
    let x, y;
    let xNext, yNext;
    let a, aNext, r;
    let max_steps = 100000;
    let steps = 0;
    let rewardSum = 0;

    if (startState) {
      [x, y] = startState;
    } else {
      [x, y] = mazeComp.getRandomStartState();
    }
    a = mazeComp.getEpsilonGreedyAction(x, y, epsilon);

    // run episode until terminal state has been reached
    while (!mazeComp.isTerminal(x, y)) {
      [xNext, yNext, r] = mazeComp.step(x, y, a);
      aNext = mazeComp.getEpsilonGreedyAction(xNext, yNext, epsilon);
      SarsaQTableUpdate(x, y, a, r, xNext, yNext, aNext);
      x = Number(xNext);
      y = Number(yNext);
      a = Number(aNext);
      rewardSum += r;

      steps++;
      if (steps >= max_steps) {
        console.log("WARNING: Maximum number of steps per episode reached.");
        stepsPerEpisode.push(steps);
        rewardPerEpisode.push(rewardSum);
        return;
      }
    }
    stepsPerEpisode.push(steps);
    rewardPerEpisode.push(rewardSum);
  };

  //====================================================
  // Expected SARSA algorithm
  //====================================================

  const ExpectedSarsaQTableUpdate = (x, y, a, r, xNext, yNext, aNext) => {
    let g;
    if (mazeComp.isTerminal(xNext, yNext)) {
      g = r;
    } else {
      let vNextExpected = 0.0;
      let prob;
      // a_next is not used to calculate the expected next state value

      // each action has a base probability to be selected of epsilon divided
      // by the number of actions (random action selection case of e-greedy)
      prob = epsilon / numA;
      for (let a = 0; a < numA; a++) {
        vNextExpected += prob * mazeComp.getQValue(xNext, yNext, a);
      }

      // the maximum Q action has a probability of (1 - epsilon)
      // to be selected (greedy action selection case of e-greedy)
      prob = 1.0 - epsilon;
      let aNextGreedy = mazeComp.getPolicy(xNext, yNext);
      vNextExpected += prob * mazeComp.getQValue(xNext, yNext, aNextGreedy);

      // G with respect to expected value V of next state
      g = r + gamma * vNextExpected;
    }
    let q = (1.0 - alpha) * mazeComp.getQValue(x, y, a) + alpha * g;
    mazeComp.setQValue(x, y, a, q);
  };

  const runExpectedSarsaEpisode = () => {
    let x, y;
    let xNext, yNext;
    let a, aNext, r;
    let max_steps = 100000;
    let steps = 0;
    let rewardSum = 0;

    if (startState) {
      [x, y] = startState;
    } else {
      [x, y] = mazeComp.getRandomStartState();
    }
    a = mazeComp.getEpsilonGreedyAction(x, y, epsilon);

    // run episode until terminal state has been reached
    while (!mazeComp.isTerminal(x, y)) {
      [xNext, yNext, r] = mazeComp.step(x, y, a);
      aNext = mazeComp.getEpsilonGreedyAction(xNext, yNext, epsilon);
      ExpectedSarsaQTableUpdate(x, y, a, r, xNext, yNext, aNext);
      x = Number(xNext);
      y = Number(yNext);
      a = Number(aNext);
      rewardSum += r;

      steps++;
      if (steps >= max_steps) {
        console.log("WARNING: Maximum number of steps per episode reached.");
        stepsPerEpisode.push(steps);
        rewardPerEpisode.push(rewardSum);
        return;
      }
    }
    stepsPerEpisode.push(steps);
    rewardPerEpisode.push(rewardSum);
  };

  //====================================================
  // Dyna-Q algorithm
  //====================================================

  let envModel = Array.from({ length: numX }, () =>
    Array.from({ length: numY }, () => Array.from({ length: numA }, () => null))
  );
  let seenStateActions = Array();

  const DynaQModelUpdate = (x, y, a, r, xNext, yNext) => {
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
      seenStateActions.push([x, y, a]);
    }
    envModel[x][y][a] = [xNext, yNext, r];
  };

  const DynaQGetModelStateAction = () => {
    let i = getRandomInt(seenStateActions.length);
    return seenStateActions[i];
  };

  const runDynaQEpisode = () => {
    let x, y;
    let xNext, yNext;
    let a, r;
    let max_steps = 100000;
    let steps = 0;
    let rewardSum = 0;

    if (startState) {
      [x, y] = startState;
    } else {
      [x, y] = mazeComp.getRandomStartState();
    }

    // run episode until terminal state has been reached
    while (!mazeComp.isTerminal(x, y)) {
      a = mazeComp.getEpsilonGreedyAction(x, y, epsilon);
      [xNext, yNext, r] = mazeComp.step(x, y, a);
      QLearningQTableUpdate(x, y, a, r, xNext, yNext);
      DynaQModelUpdate(x, y, a, r, xNext, yNext);
      x = Number(xNext);
      y = Number(yNext);
      rewardSum += r;

      // planning steps (model based Q table update steps)
      for (let n = 0; n < planningSteps; n++) {
        let mx, my, ma, mr;
        let mxNext, myNext;
        [mx, my, ma] = DynaQGetModelStateAction();
        [mxNext, myNext, mr] = envModel[mx][my][ma];
        QLearningQTableUpdate(mx, my, ma, mr, mxNext, myNext);
      }

      steps++;
      if (steps >= max_steps) {
        console.log("WARNING: Maximum number of steps per episode reached.");
        stepsPerEpisode.push(steps);
        rewardPerEpisode.push(rewardSum);
        return;
      }
    }
    stepsPerEpisode.push(steps);
    rewardPerEpisode.push(rewardSum);
  };

  //====================================================
  // Monte Carlo Algorithm
  //====================================================

  let trajectory;

  const MonteCarloQTableUpdate = () => {
    for (let y = 0; y < numY; y++) {
      for (let x = 0; x < numX; x++) {
        for (let a = 0; a < numA; a++) {
          let take = false;
          let g = 0.0;
          let gammaProduct = 1.0;

          for (let n = 0; n < trajectory.length; n++) {
            if (
              x == trajectory[n][0] &&
              y == trajectory[n][1] &&
              a == trajectory[n][2]
            ) {
              take = true;
            }
            if (take) {
              g += gammaProduct * trajectory[n][3];
              gammaProduct *= gamma;
            }
          }

          if (take) {
            // incremental average Q value update
            let q = (1.0 - alpha) * mazeComp.getQValue(x, y, a) + alpha * g;
            mazeComp.setQValue(x, y, a, q);
          }
        }
      }
    }
  };

  const runMonteCarloEpisode = () => {
    let x, y;
    let xNext, yNext;
    let a, r;
    let max_steps = 100000;
    let steps = 0;
    let rewardSum = 0;

    [x, y] = mazeComp.getRandomStartState();
    trajectory = Array();

    // run episode until terminal state has been reached
    while (!mazeComp.isTerminal(x, y)) {
      a = mazeComp.getEpsilonGreedyAction(x, y, epsilon);
      [xNext, yNext, r] = mazeComp.step(x, y, a);
      trajectory.push([x, y, a, r]);
      x = Number(xNext);
      y = Number(yNext);
      rewardSum += r;

      steps++;
      if (steps >= max_steps) {
        console.log("WARNING: Maximum number of steps per episode reached.");
        stepsPerEpisode.push(steps);
        rewardPerEpisode.push(rewardSum);
        return;
      }
    }
    MonteCarloQTableUpdate();
    stepsPerEpisode.push(steps);
    rewardPerEpisode.push(rewardSum);
  };

  //====================================================

  const runEpisode = () => {
    if (algorithm == "Q-Learning") {
      runQLearningEpisode();
    }
    if (algorithm == "SARSA") {
      runSarsaEpisode();
    }
    if (algorithm == "Expected SARSA") {
      runExpectedSarsaEpisode();
    }
    if (algorithm == "Dyna-Q") {
      runDynaQEpisode();
    }
    if (algorithm == "Monte Carlo") {
      runMonteCarloEpisode();
    }
    plotComp.updatePlot();
  };

  //====================================================

  const run = () => {
    timer = setTimeout(() => {
      if (episode < numEpisodes) {
        episode++;
        runEpisode();
        run();
      }
    }, 100);
  };

  const halt = () => {
    if (timer) {
      clearTimeout(timer);
    }
  };

  const init = () => {
    episode = 0;
    stepsPerEpisode = [];
    rewardPerEpisode = [];
    mazeComp.initQValues();
    plotComp.clearPlot();
  };
</script>

<style>
  div.container {
    margin: 20px auto;
    padding: 10px;
    border: 3px solid #eee;
  }
  div.box {
    display: flex;
    justify-content: center;
    margin: 20px;
  }
  select {
    margin: 0px 20px;
    color: inherit;
    background: inherit;
    padding: 0px 8px;
  }
  div.narrow-box {
    margin: 0px;
  }
  button {
    margin: 0px 20px;
  }
</style>

<div class="container" style="width: {numX * 100 + (numX - 1) * 4}px;">
  <div class="narrow-box">
    <LinePlot
      bind:this={plotComp}
      bind:data={stepsPerEpisode}
      bind:dataSecond={rewardPerEpisode}
      yTitle={'steps per episode'}
      yIsLog={true}
      ySecondTitle={'reward per episode'}
      hasSecondY={true} />
  </div>

  <div class="box">EPISODE : {episode}</div>
  <div class="box">
    <select bind:value={algorithm} on:change={init}>
      <option value="Q-Learning">Q-Learning</option>
      <option value="SARSA">SARSA</option>
      <option value="Expected SARSA">Expected SARSA</option>
      <option value="Dyna-Q">Dyna-Q</option>
      <option value="Monte Carlo">Monte Carlo</option>
    </select>
    <button on:click={init}>init</button>
    <button on:click={halt}>halt</button>
    <button on:click={run}>run</button>
  </div>

  <div class="narrow-box">
    <Maze
      bind:numX
      bind:numY
      bind:blocked
      bind:terminal
      bind:rewards
      bind:defaultReward
      bind:this={mazeComp} />
  </div>
</div>
