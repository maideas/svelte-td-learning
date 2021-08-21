<script>
  import { onMount } from "svelte";

  let model;

  onMount(() => {
    model = createModel();
  });

  const createModel = () => {
    const model = tf.sequential();
    const lr = 0.005;

    model.add(
      tf.layers.dense({
        inputShape: [2],
        units: 10,
        useBias: true,
        activation: "tanh"
      })
    );
    model.add(
      tf.layers.dense({ units: 10, useBias: true, activation: "tanh" })
    );
    model.add(
      tf.layers.dense({ units: 10, useBias: true, activation: "tanh" })
    );
    model.add(
      tf.layers.dense({ units: 4, useBias: true, activation: "linear" })
    );

    model.compile({
      optimizer: tf.train.adam(lr),
      loss: "meanSquaredError",
      metrics: ["accuracy"]
    });
    return model;
  };

  export const resetModel = () => {
    model.resetStates();
  }

  export const fit = (dataX, dataY) => {
    const epochs = 10;

    const trainX = tf.tensor2d(
      dataX,
      [dataX.length, dataX[0].length],
      "float32"
    );
    const trainY = tf.tensor2d(
      dataY,
      [dataY.length, dataY[0].length],
      "float32"
    );

    const onEpochEnd = (n, logs) => {
      console.log("onEpochEnd", "[ epoch", n, "of", epochs, "]", logs);
    };

    return model.fit(trainX, trainY, {
      batchSize: dataX.length,
      epochs: epochs,
      shuffle: true,
      callbacks: {
        /* onEpochEnd */
      }
    });
  };

  export const predict = dataX => {
    const X = tf.tensor2d(dataX, [dataX.length, dataX[0].length], "float32");
    return model.predict(X).array();
  };

  export const normalize = (dataX, min, max) => {
    let dataN = [];
    dataX.forEach((x, i) => {
      x -= (max[i] + min[i]) / 2.0;
      x /= (max[i] - min[i]) / 2.0;
      dataN.push(x);
    });
    return dataN;
  };
</script>
