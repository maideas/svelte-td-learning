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

  export const isBlocked = (x, y) => {
    return mazeTileComps[x][y].isBlocked();
  };

  export const isTerminal = (x, y) => {
    return mazeTileComps[x][y].isTerminal();
  };

  export const initQValues = () => {
    for (let y = 0; y < numY; y++) {
      for (let x = 0; x < numX; x++) {
        mazeTileComps[x][y].initQValues();
      }
    }
    updateHeatMap();
  };

  export const setQValue = (x, y, a, val) => {
    if (x >= 0 && x < numX && y >= 0 && y < numY) {
      mazeTileComps[x][y].setQValue(a, val);
      updateHeatMap();
    } else {
      console.log("ERROR: Invalid setQValue coordinates [", x, ":", y, "] !");
    }
  };

  export const getQValue = (x, y, a) => {
    return mazeTileComps[x][y].getQValue(a);
  };

  export const getMaxQValue = (x, y) => {
    return mazeTileComps[x][y].getMaxQValue();
  };

  export const getPolicy = (x, y) => {
    return mazeTileComps[x][y].getPolicy();
  };

  const getReward = (x, y) => {
    if (x >= 0 && x < numX && y >= 0 && y < numY) {
      return mazeTileComps[x][y].getReward();
    } else {
      console.log("ERROR: Invalid getReward coordinates [", x, ":", y, "] !");
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
      let x = getRandomInt(numX);
      let y = getRandomInt(numY);
      if (!isTerminal(x, y) && !isBlocked(x, y)) {
        return [x, y];
      }
    }
  };

  export const getEpsilonGreedyAction = (x, y, epsilon) => {
    if (Math.random() < epsilon) {
      return getRandomInt(numA); // choose random action with epsilon probability
    } else {
      return getPolicy(x, y); // else choose action according to current policy
    }
  };

  export const step = (x, y, a) => {
    let xNext = Number(x);
    let yNext = Number(y);

    if (a == MazeDirEnum.down && y < numY - 1) {
      yNext += 1;
    }
    if (a == MazeDirEnum.right && x < numX - 1) {
      xNext += 1;
    }
    if (a == MazeDirEnum.up && y > 0) {
      yNext -= 1;
    }
    if (a == MazeDirEnum.left && x > 0) {
      xNext -= 1;
    }
    if (isBlocked(xNext, yNext)) {
      xNext = Number(x);
      yNext = Number(y);
    }
    // in this example environment immediate reward is only based on next state value
    return [xNext, yNext, getReward(xNext, yNext)];
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
