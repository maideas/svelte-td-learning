<script>
  import { onMount } from "svelte";
  import { createEventDispatcher } from "svelte";

  const dispatch = createEventDispatcher();

  export let numX;
  export let numY;
  export let numA;

  let logits = Array.from({ length: numX }, () =>
    Array.from({ length: numY }, () => null)
  );

  let targetLogits = Array.from({ length: numX }, () =>
    Array.from({ length: numY }, () => null)
  );

  //====================================================
  // init
  //====================================================

  export const init = () => {
    for (let y = 0; y < numY; y++) {
      for (let x = 0; x < numX; x++) {
        logits[x][y] = Array(numA)
          .fill()
          .map(() => Math.random());
        targetLogits[x][y] = [...logits[x][y]];
      }
    }

    // notify the upper components, that the model has changed ...
    dispatch("modelChanged", {});
  };

  onMount(() => {
    init();
  });

  //====================================================
  // predict
  //====================================================

  export const predictState = state => {
    let x = state[0];
    let y = state[1];

    return logits[x][y];
  };

  //====================================================
  // update
  //====================================================

  export const updateModel = (stepData, calcLogitsFunc) => {
    let x = stepData.state[0];
    let y = stepData.state[1];
    let a = stepData.a;

    // "filters" the logits table update data
    const alpha = 0.25;  // (0..1]

    targetLogits[x][y][a] =
      (1 - alpha) * targetLogits[x][y][a] +
      alpha * calcLogitsFunc(logits[x][y], stepData)[a];
  };

  export const takeModel = () => {
    for (let y = 0; y < numY; y++) {
      for (let x = 0; x < numX; x++) {
        logits[x][y] = [...targetLogits[x][y]];
      }
    }

    // notify the upper components, that the model has changed ...
    dispatch("modelChanged", {});
  };
</script>
