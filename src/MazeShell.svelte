<script>
  import { onMount } from "svelte";
  import Maze from "./Maze.svelte";
  import { numA } from "./MazeTile.svelte";
  import LinePlot from "./LinePlot.svelte";
  import Data from "./Data.svelte";
  import QNet from "./QNet.svelte";

  const epsilon = 0.1; // exploration probability
  const alpha = 0.2; // learning rate
  const gamma = 0.9; // future reward discount factor

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

  const numStates = numX * numY;

  //====================================================

  let mazeComp;
  let plotComp;
  let stepsPerEpisode = [];
  let rewardPerEpisode = [];
  let rewardSum = 0;
  let selectedAlgorithm;

  let episodeTimer;
  let episode = 0;
  let stepTimer;
  let steps = 0;

  //====================================================

  const maxData = 2000;
  const batchSize = 100;
  const trainDistance = 10;

  let MazeDataComp;
  let DataComp;
  let QNetComp;
  let useQNet = false;
  let trainDistanceCount = 0;

  //====================================================

  const normalizeState = state => {
    return QNetComp.normalize(state, [0, 0], [numX - 1, numY - 1]);
  };

  onMount(() => {
    for (let y = 0; y < numY; y++) {
      for (let x = 0; x < numX; x++) {
        let state = [x, y];
        let normState = normalizeState(state);
        MazeDataComp.add({ state, normState });
      }
    }
  });

  //====================================================
  // Q table update
  //====================================================

  const QTableUpdate = (stepData, calcQValueFunc) => {
    let q = calcQValueFunc(stepData);
    mazeComp.setQValue(stepData.state, stepData.a, q);
  };

  //====================================================
  // Q network update
  //====================================================

  const QNetUpdate = (stepData, calcQValueFunc) => {
    // normalize state value and add it to stepData item ...
    stepData.normState = normalizeState(stepData.state);

    // add the given data item to the DataComp memory ...
    DataComp.add(stepData);

    // train Q network only every "trainDistance" steps ...
    trainDistanceCount++;
    if (trainDistanceCount < trainDistance) return;
    trainDistanceCount = 0;

    // get a random batch of data from the DataComp memory ...
    const stepDataBatch = DataComp.getBatch(batchSize);

    // prepare Q network input data (normalized state values) ...
    let normStates = [];
    stepDataBatch.forEach(stepData => {
      normStates.push(stepData.normState);
    });

    // get current Q network output data (Q values) for the
    // given input data (states) ...
    let QValues = QNetComp.predict(normStates);

    stepDataBatch.forEach((stepData, i) => {
      // update the selected actions related Q values ...
      QValues[i][stepData.a] = calcQValueFunc(stepData);
    });

    // use the prepared X and Y data to adjust the Q network ...
    QNetComp.fit(normStates, QValues);

    // mazeComp is used to query Q values, so we need to:
    // update all mazeComp Q values using the adjusted Q network ...
    MazeDataComp.getAll().forEach(data => {
      let QValues = QNetComp.predict([data.normState]);
      mazeComp.setQValues(data.state, QValues[0]);
    });

    // tensorflow memory footprint debugging ...
    //console.table(tf.memory());
  };

  //====================================================
  // Q-Learning algorithm
  //====================================================

  const QLearningCalcQValue = stepData => {
    let g;
    if (mazeComp.isTerminal(stepData.stateNext)) {
      g = stepData.r;
    } else {
      // G with respect to maximum Q value (over all actions) of next state
      g = stepData.r + gamma * mazeComp.getMaxQValue(stepData.stateNext);
    }
    if (useQNet) {
      // pure g is returned in case a DQN is used, because the network
      // learning rate replaces the table variant alpha parameter ...
      return g;
    }
    return (
      (1.0 - alpha) * mazeComp.getQValue(stepData.state, stepData.a) + alpha * g
    );
  };

  const runQLearningEpisodeStep = state => {
    let stateNext;
    let a, r;

    if (mazeComp.isTerminal(state)) {
      stepsPerEpisode.push(steps);
      rewardPerEpisode.push(rewardSum);
      plotComp.updatePlot();
      runEpisode();
    } else {
      stepTimer = setTimeout(() => {
        a = mazeComp.getEpsilonGreedyAction(state, epsilon);
        [stateNext, r] = mazeComp.step(state, a);
        if (useQNet) {
          QNetUpdate({ state, a, r, stateNext }, QLearningCalcQValue);
        } else {
          QTableUpdate({ state, a, r, stateNext }, QLearningCalcQValue);
        }
        state = [...stateNext];

        rewardSum += r;
        steps++;
        runQLearningEpisodeStep(state);
      }, 0);
    }
  };

  const runQLearningEpisode = () => {
    let state;

    rewardSum = 0;
    steps = 0;

    if (startState) {
      state = startState;
    } else {
      state = mazeComp.getRandomStartState();
    }
    runQLearningEpisodeStep(state);
  };

  //====================================================
  // SARSA algorithm
  //====================================================

  const SarsaCalcQValue = stepData => {
    let g;
    if (mazeComp.isTerminal(stepData.stateNext)) {
      g = stepData.r;
    } else {
      // G with respect to policy related Q value of next state
      g =
        stepData.r +
        gamma * mazeComp.getQValue(stepData.stateNext, stepData.aNext);
    }
    if (useQNet) {
      // pure g is returned in case a DQN is used, because the network
      // learning rate replaces the table variant alpha parameter ...
      return g;
    }
    return (
      (1.0 - alpha) * mazeComp.getQValue(stepData.state, stepData.a) + alpha * g
    );
  };

  const runSarsaEpisodeStep = (state, a) => {
    let stateNext;
    let aNext;
    let r;

    if (mazeComp.isTerminal(state)) {
      stepsPerEpisode.push(steps);
      rewardPerEpisode.push(rewardSum);
      plotComp.updatePlot();
      runEpisode();
    } else {
      stepTimer = setTimeout(() => {
        [stateNext, r] = mazeComp.step(state, a);
        aNext = mazeComp.getEpsilonGreedyAction(stateNext, epsilon);
        if (useQNet) {
          QNetUpdate({ state, a, r, stateNext, aNext }, SarsaCalcQValue);
        } else {
          QTableUpdate({ state, a, r, stateNext, aNext }, SarsaCalcQValue);
        }
        state = [...stateNext];
        a = Number(aNext);

        rewardSum += r;
        steps++;
        runSarsaEpisodeStep(state, a);
      }, 0);
    }
  };

  const runSarsaEpisode = () => {
    let state;
    let a;

    rewardSum = 0;
    steps = 0;

    if (startState) {
      state = startState;
    } else {
      state = mazeComp.getRandomStartState();
    }
    a = mazeComp.getEpsilonGreedyAction(state, epsilon);
    runSarsaEpisodeStep(state, a);
  };

  //====================================================
  // Expected SARSA algorithm
  //====================================================

  const ExpectedSarsaCalcQValue = stepData => {
    let g;
    if (mazeComp.isTerminal(stepData.stateNext)) {
      g = stepData.r;
    } else {
      let vNextExpected = 0.0;
      let prob;
      // aNext is not used to calculate the expected next state value

      // each action has a base probability to be selected of epsilon divided
      // by the number of actions (random action selection case of e-greedy)
      prob = epsilon / numA;
      for (let a = 0; a < numA; a++) {
        vNextExpected += prob * mazeComp.getQValue(stepData.stateNext, a);
      }

      // the maximum Q action has a probability of (1 - epsilon)
      // to be selected (greedy action selection case of e-greedy)
      prob = 1.0 - epsilon;
      let aNextGreedy = mazeComp.getPolicy(stepData.stateNext);
      vNextExpected +=
        prob * mazeComp.getQValue(stepData.stateNext, aNextGreedy);

      // G with respect to expected value V of next state
      g = stepData.r + gamma * vNextExpected;
    }
    if (useQNet) {
      // pure g is returned in case a DQN is used, because the network
      // learning rate replaces the table variant alpha parameter ...
      return g;
    }
    return (
      (1.0 - alpha) * mazeComp.getQValue(stepData.state, stepData.a) + alpha * g
    );
  };

  const runExpectedSarsaEpisodeStep = (state, a) => {
    let stateNext;
    let aNext, r;

    if (mazeComp.isTerminal(state)) {
      stepsPerEpisode.push(steps);
      rewardPerEpisode.push(rewardSum);
      plotComp.updatePlot();
      runEpisode();
    } else {
      stepTimer = setTimeout(() => {
        [stateNext, r] = mazeComp.step(state, a);
        aNext = mazeComp.getEpsilonGreedyAction(stateNext, epsilon);
        if (useQNet) {
          QNetUpdate(
            { state, a, r, stateNext, aNext },
            ExpectedSarsaCalcQValue
          );
        } else {
          QTableUpdate(
            { state, a, r, stateNext, aNext },
            ExpectedSarsaCalcQValue
          );
        }
        state = [...stateNext];
        a = Number(aNext);

        rewardSum += r;
        steps++;
        runExpectedSarsaEpisodeStep(state, a);
      }, 0);
    }
  };

  const runExpectedSarsaEpisode = () => {
    let state;
    let a;

    rewardSum = 0;
    steps = 0;

    if (startState) {
      state = startState;
    } else {
      state = mazeComp.getRandomStartState();
    }
    a = mazeComp.getEpsilonGreedyAction(state, epsilon);
    runExpectedSarsaEpisodeStep(state, a);
  };

  //====================================================
  // Dyna-Q algorithm
  //====================================================

  let envModel = Array.from({ length: numX }, () =>
    Array.from({ length: numY }, () => Array.from({ length: numA }, () => null))
  );
  let seenStateActions = [];

  const DynaQModelUpdate = (state, a, r, stateNext) => {
    let x = state[0];
    let y = state[1];
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
      seenStateActions.push([state, a]);
    }
    envModel[x][y][a] = [stateNext, r];
  };

  const DynaQGetModelStateAction = () => {
    let i = mazeComp.getRandomInt(seenStateActions.length);
    return seenStateActions[i];
  };

  const runDynaQEpisodeStep = state => {
    let stateNext;
    let a, r;

    if (mazeComp.isTerminal(state)) {
      stepsPerEpisode.push(steps);
      rewardPerEpisode.push(rewardSum);
      plotComp.updatePlot();
      runEpisode();
    } else {
      stepTimer = setTimeout(() => {
        a = mazeComp.getEpsilonGreedyAction(state, epsilon);
        [stateNext, r] = mazeComp.step(state, a);
        QTableUpdate({ state, a, r, stateNext }, QLearningCalcQValue);
        DynaQModelUpdate(state, a, r, stateNext);
        state = [...stateNext];

        // planning steps (model based Q table update steps)
        for (let n = 0; n < planningSteps; n++) {
          let mState;
          let ma, mr;
          let mStateNext;
          [mState, ma] = DynaQGetModelStateAction();
          [mStateNext, mr] = envModel[mState[0]][mState[1]][ma];
          QTableUpdate(
            { state: mState, a: ma, r: mr, stateNext: mStateNext },
            QLearningCalcQValue
          );
        }

        rewardSum += r;
        steps++;
        runDynaQEpisodeStep(state);
      }, 0);
    }
  };

  const runDynaQEpisode = () => {
    let state;
    let a;

    rewardSum = 0;
    steps = 0;

    if (startState) {
      state = startState;
    } else {
      state = mazeComp.getRandomStartState();
    }
    runDynaQEpisodeStep(state);
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
              x == trajectory[n][0][0] &&
              y == trajectory[n][0][1] &&
              a == trajectory[n][1]
            ) {
              take = true;
            }
            if (take) {
              g += gammaProduct * trajectory[n][2];
              gammaProduct *= gamma;
            }
          }

          if (take) {
            // incremental average Q value update
            let q = (1.0 - alpha) * mazeComp.getQValue([x, y], a) + alpha * g;
            mazeComp.setQValue([x, y], a, q);
          }
        }
      }
    }
  };

  const runMonteCarloEpisodeStep = state => {
    let stateNext;
    let a, r;

    // run episode until terminal state has been reached
    if (mazeComp.isTerminal(state)) {
      MonteCarloQTableUpdate();
      stepsPerEpisode.push(steps);
      rewardPerEpisode.push(rewardSum);
      plotComp.updatePlot();
      runEpisode();
    } else {
      stepTimer = setTimeout(() => {
        a = mazeComp.getEpsilonGreedyAction(state, epsilon);
        [stateNext, r] = mazeComp.step(state, a);
        trajectory.push([state, a, r]);
        state = [...stateNext];

        rewardSum += r;
        steps++;
        runMonteCarloEpisodeStep(state);
      }, 0);
    }
  };

  const runMonteCarloEpisode = () => {
    let state;

    rewardSum = 0;
    steps = 0;
    trajectory = [];

    state = mazeComp.getRandomStartState();
    runMonteCarloEpisodeStep(state);
  };

  //====================================================

  const algorithms = [
    { name: "Q-Learning", func: runQLearningEpisode },
    { name: "SARSA", func: runSarsaEpisode },
    { name: "Expected-SARSA", func: runExpectedSarsaEpisode },
    { name: "Dyna-Q", func: runDynaQEpisode },
    { name: "Monte Carlo", func: runMonteCarloEpisode }
  ];

  const runEpisode = () => {
    episodeTimer = setTimeout(() => {
      if (episode < numEpisodes) {
        episode++;
        algorithms.forEach(algo => {
          if (selectedAlgorithm == algo.name) {
            algo.func();
          }
        });
      }
    }, 0);
  };

  const halt = () => {
    if (episodeTimer) {
      clearTimeout(episodeTimer);
    }
    if (stepTimer) {
      clearTimeout(stepTimer);
    }
  };

  const init = () => {
    episode = 0;
    stepsPerEpisode = [];
    rewardPerEpisode = [];
    mazeComp.initQValues();
    plotComp.clearPlot();
    QNetComp.resetModel();
    DataComp.clear();
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
    margin: 0px 10px;
    color: inherit;
    background: inherit;
  }
  div.narrow-box {
    margin: 0px;
  }
  button {
    margin: 0px 10px;
  }
  input {
    margin: 5px;
  }
  div.flex {
    display: flex;
    align-items: center;
    margin: 0 10px;
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
    <div class="flex">
      <input type="checkbox" bind:checked={useQNet} on:change={init} />
      DQN
    </div>
    <button on:click={init}>init</button>
    <button on:click={halt}>halt</button>
    <button on:click={runEpisode}>run</button>
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

<Data maxData={numStates} bind:this={MazeDataComp} />
<Data {maxData} bind:this={DataComp} />
<QNet bind:this={QNetComp} />
