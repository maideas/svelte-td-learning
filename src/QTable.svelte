<script>
  import { onMount } from "svelte";
  import { createEventDispatcher } from "svelte";

  const dispatch = createEventDispatcher();

  export let numX;
  export let numY;
  export let numA;

  let QValues = Array.from({ length: numX }, () =>
    Array.from({ length: numY }, () => null)
  );

  //====================================================
  // init
  //====================================================

  export const init = () => {
    for (let y = 0; y < numY; y++) {
      for (let x = 0; x < numX; x++) {
        QValues[x][y] = Array(numA)
          .fill()
          .map(() => Math.random());
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

    return QValues[x][y];
  };

  //====================================================
  // update
  //====================================================

  export const updateModel = (stepData, calcQValueFunc) => {
    let x = stepData.state[0];
    let y = stepData.state[1];
    let a = stepData.a;

    QValues[x][y][a] = calcQValueFunc(stepData);

    // notify the upper components, that the model has changed ...
    dispatch("modelChanged", {});
  };
</script>
