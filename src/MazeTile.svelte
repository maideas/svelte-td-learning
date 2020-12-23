<script>
  import { onMount } from "svelte";
  import Icon from "fa-svelte";
  import { faArrowAltCircleUp } from "@fortawesome/free-regular-svg-icons";

  const MazeDirEnum = Object.freeze({
    up: 0,
    right: 1,
    down: 2,
    left: 3
  });

  let blocked = false;
  let terminal = false;
  let reward = 0;
  let QValues;

  let maxQValue;
  let policy;
  let angle;
  let bg_style = "";
  let negReward;

  $: {
    maxQValue = QValues == undefined ? 1.0 : Math.max(...QValues);
    policy =
      QValues == undefined
        ? MazeDirEnum.up
        : QValues.indexOf(Math.max(...QValues));
    angle = (policy * 90).toFixed();
    negReward = reward < 0.0;
  }

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
  };

  export const getPolicy = () => {
    return policy;
  };

  export const getQValue = dir => {
    return terminal ? reward : QValues[dir];
  };

  export const getMaxQValue = () => {
    return terminal ? reward : maxQValue;
  };

  export const setQValue = (dir, val) => {
    QValues[dir] = val;
  };

  export const initQValues = () => {
    QValues = Array(4)
      .fill()
      .map(() => Math.random());
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

  initQValues();
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
  .blocked span,
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
  .negReward {
    color: #a00;
  }
  .sub-tile-reward span {
    display: inline;
  }
</style>

<div class="tile" class:blocked class:terminal style={bg_style}>
  <div />
  <div class="sub-tile sub-tile-top">
    <span>{QValues[MazeDirEnum.up].toFixed(3)}</span>
  </div>
  <div class="sub-tile sub-tile-reward" class:negReward>
    {#if Math.abs(reward) < 10.0}
      <span>{reward.toFixed(1)}</span>
    {:else}
      <span>{reward.toFixed()}</span>
    {/if}
  </div>

  <div class="sub-tile sub-tile-left">
    <span>{QValues[MazeDirEnum.left].toFixed(3)}</span>
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
    <span>{QValues[MazeDirEnum.right].toFixed(3)}</span>
  </div>

  <div />
  <div class="sub-tile sub-tile-bottom">
    <span>{QValues[MazeDirEnum.down].toFixed(3)}</span>
  </div>
  <div />
</div>
