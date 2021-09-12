<script context="module">
  export const MazeDirEnum = Object.freeze({
    up: 0,
    right: 1,
    down: 2,
    left: 3
  });

  // number of possible actions (= number of directions)
  export const numA = 4;
</script>

<script>
  import { onMount } from "svelte";
  import Icon from "fa-svelte";
  import { faArrowAltCircleUp } from "@fortawesome/free-regular-svg-icons";

  let blocked = false;
  let terminal = false;
  let reward = 0;
  let values = Array.from({ length: numA }, () => 0);

  let angle;
  let bg_style = "";

  export const isBlocked = () => {
    return blocked;
  };
  export const setBlocked = () => {
    blocked = true;
  };

  export const isTerminal = () => {
    return terminal;
  };
  export const setTerminal = () => {
    terminal = true;
  };

  export const getReward = () => {
    return reward;
  };
  export const setReward = r => {
    reward = r;
    if (terminal) {
      values = Array.from({ length: numA }, () => r);
    }
  };

  export const getMaxValue = () => {
    return Math.max(...values);
  };

  const updatePolicy = () => {
    let policy = values.indexOf(getMaxValue());
    angle = (policy * 90).toFixed();
  };

  export const setValues = vs => {
    if (!terminal && !blocked) {
      values = vs;
      updatePolicy();
    }
  };

  export const setHeat = h => {
    let bg_r, bg_g, bg_b;
    let heat = 0.35 + 0.6 * h;

    if (!blocked) {
      bg_r = 0 + heat * 255;
      bg_g = 110 + heat * 145;
      bg_b = 210 + heat * 45;
      bg_style = "background: rgb(" + bg_r + "," + bg_g + "," + bg_b + ", 1);";
    }
  };
</script>

<style>
  .tile {
    display: grid;
    grid-template-columns: 25px 50px 25px;
    grid-template-rows: 25px 50px 25px;
    font-size: 14px;
    width: 100px;
  }
  .sub-tile {
    display: flex;
    justify-content: center;
    align-items: center;
    white-space: nowrap;
  }
  .sub-tile-left span,
  .sub-tile-right span {
    transform: rotate(-90deg);
  }
  .blocked {
    background: #eee;
  }
  .terminal span {
    display: none;
  }
  .sub-tile-middle {
    font-size: 16px;
    line-height: 1.5em;
    white-space: normal;
    text-align: center;
  }
  .sub-tile-middle div {
    width: 32px;
    height: 32px;
    border-radius: 50%;
  }
  .sub-tile-reward {
    display: flex;
    justify-content: right;
    align-items: center;
    white-space: nowrap;
    font-size: 0.85em;
    padding-right: 3px;
    color: #0a0;
  }
  .reward-is-negative {
    color: #a00;
  }
  .sub-tile-reward span {
    display: inline;
  }
  .blocked span {
    display: none;
  }
</style>

<div class="tile" class:blocked class:terminal style={bg_style}>
  <div />
  <div class="sub-tile sub-tile-top">
    <span>{values[MazeDirEnum.up].toFixed(3)}</span>
  </div>
  <div class="sub-tile sub-tile-reward" class:reward-is-negative={reward < 0.0}>
    {#if Math.abs(reward) < 10.0}
      <span>{reward.toFixed(1)}</span>
    {:else}
      <span>{reward.toFixed()}</span>
    {/if}
  </div>

  <div class="sub-tile sub-tile-left">
    <span>{values[MazeDirEnum.left].toFixed(3)}</span>
  </div>
  <div class="sub-tile sub-tile-middle">
    {#if blocked}
      blocked
    {:else if terminal}
      terminal state
    {:else}
      <div style="font-size: 32px; transform: rotate({angle}deg);">
        <Icon icon={faArrowAltCircleUp} />
      </div>
    {/if}
  </div>
  <div class="sub-tile sub-tile-right">
    <span>{values[MazeDirEnum.right].toFixed(3)}</span>
  </div>

  <div />
  <div class="sub-tile sub-tile-bottom">
    <span>{values[MazeDirEnum.down].toFixed(3)}</span>
  </div>
  <div />
</div>
