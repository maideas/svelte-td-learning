<script>
  import { onMount } from "svelte";
  import MazeTile from "./MazeTile.svelte";
  import { MazeDirEnum, numA } from "./MazeTile.svelte";

  export let numX;
  export let numY;
  export let blocked = [];
  export let terminal = [];
  export let rewards = [];
  export let defaultReward = 0;

  let mazeTileComps = Array.from({ length: numX }, () =>
    Array.from({ length: numY }, () => null)
  );

  onMount(() => {
    blocked.forEach(coord => {
      mazeTileComps[coord[0]][coord[1]].setBlocked();
    });
    terminal.forEach(coord => {
      mazeTileComps[coord[0]][coord[1]].setTerminal();
    });

    for (let y = 0; y < numY; y++) {
      for (let x = 0; x < numX; x++) {
        mazeTileComps[x][y].setReward(defaultReward);
      }
    }
    rewards.forEach(data => {
      mazeTileComps[data[0]][data[1]].setReward(data[2]);
    });
    initQValues();
  });

  //====================================================

  export const getRandomInt = n => {
    // result range [0 .. n-1]
    return Math.floor(Math.random() * Math.floor(n));
  };

  //====================================================

  export const isBlocked = state => {
    return mazeTileComps[state[0]][state[1]].isBlocked();
  };

  export const isTerminal = state => {
    return mazeTileComps[state[0]][state[1]].isTerminal();
  };

  export const initQValues = () => {
    for (let y = 0; y < numY; y++) {
      for (let x = 0; x < numX; x++) {
        mazeTileComps[x][y].initQValues();
      }
    }
    updateHeatMap();
  };

  export const setQValue = (state, a, val) => {
    if (state[0] >= 0 && state[0] < numX && state[1] >= 0 && state[1] < numY) {
      mazeTileComps[state[0]][state[1]].setQValue(a, val);
      updateHeatMap();
    } else {
      console.log(
        "ERROR: Invalid setQValue coordinates [",
        state[0],
        ":",
        state[1],
        "] !"
      );
    }
  };

  export const getQValue = (state, a) => {
    return mazeTileComps[state[0]][state[1]].getQValue(a);
  };

  export const getMaxQValue = state => {
    return mazeTileComps[state[0]][state[1]].getMaxQValue();
  };

  export const getPolicy = state => {
    return mazeTileComps[state[0]][state[1]].getPolicy();
  };

  const getReward = state => {
    if (state[0] >= 0 && state[0] < numX && state[1] >= 0 && state[1] < numY) {
      return mazeTileComps[state[0]][state[1]].getReward();
    } else {
      console.log(
        "ERROR: Invalid getReward coordinates [",
        state[0],
        ":",
        state[1],
        "] !"
      );
      return 0;
    }
  };

  const updateHeatMap = () => {
    let minValue = 1000000;
    let maxValue = -1000000;

    for (let y = 0; y < numY; y++) {
      for (let x = 0; x < numX; x++) {
        let value = mazeTileComps[x][y].getMaxQValue();
        if (minValue > value) {
          minValue = value;
        }
        if (maxValue < value) {
          maxValue = value;
        }
      }
    }
    let delta = maxValue - minValue;

    for (let y = 0; y < numY; y++) {
      for (let x = 0; x < numX; x++) {
        let value = mazeTileComps[x][y].getMaxQValue();
        mazeTileComps[x][y].setHeat((value - minValue) / delta);
      }
    }
  };

  //====================================================

  export const getRandomStartState = () => {
    while (true) {
      let state = [getRandomInt(numX), getRandomInt(numY)];
      if (!isTerminal(state) && !isBlocked(state)) {
        return state;
      }
    }
  };

  export const getEpsilonGreedyAction = (state, epsilon) => {
    if (Math.random() < epsilon) {
      return getRandomInt(numA); // choose random action with epsilon probability
    } else {
      return getPolicy(state); // else choose action according to current policy
    }
  };

  export const step = (state, a) => {
    let stateNext = [...state];

    if (a == MazeDirEnum.down && state[1] < numY - 1) {
      stateNext[1] += 1;
    }
    if (a == MazeDirEnum.right && state[0] < numX - 1) {
      stateNext[0] += 1;
    }
    if (a == MazeDirEnum.up && state[1] > 0) {
      stateNext[1] -= 1;
    }
    if (a == MazeDirEnum.left && state[0] > 0) {
      stateNext[0] -= 1;
    }
    if (isBlocked(stateNext)) {
      stateNext = [...state];
    }
    // in this example environment immediate reward is only based on next state value
    return [stateNext, getReward(stateNext)];
  };
</script>

<style>
  .maze {
    display: grid;
    grid-gap: 4px;
    justify-content: center;
  }
</style>

<div class="maze" style="grid-template-columns: repeat({numX}, 100px);">
  {#each Array(numY) as _, y}
    {#each Array(numX) as _, x}
      <MazeTile bind:this={mazeTileComps[x][y]} />
    {/each}
  {/each}
</div>
