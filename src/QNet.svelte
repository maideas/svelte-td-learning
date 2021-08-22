<script>
  import { onMount } from "svelte";

  export let duelingQNet = false;

  class DuelingLayer extends tf.layers.Layer {
    constructor() {
      super({});
    }
    getClassName() {
      return "DuelingLayer";
    }
    computeOutputShape(inputShape) {
      return inputShape[0];
    }
    call(input, kwargs) {
      return tf.tidy(() => {
        const A = input[0];
        const V = input[1];
        const axis = 1;
        return A.sub(A.mean(axis).reshape([-1, 1])).add(V);
      });
    }
  }

  let model = undefined;

  onMount(() => {
    initModel();
  });

  export const initModel = () => {
    const lr = 0.005;

    if (model != undefined) {
      model.dispose();
    }

    if (duelingQNet) {
      createDuelingModel();
    } else {
      createModel();
    }

    model.compile({
      optimizer: tf.train.adam(lr),
      loss: "meanSquaredError",
      metrics: ["accuracy"]
    });
  };

  const createModel = () => {
    model = tf.sequential();

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
  };

  const createDuelingModel = () => {
    const input = tf.input({ shape: [2] });

    const dense1 = tf.layers
      .dense({ units: 10, useBias: true, activation: "tanh" })
      .apply(input);
    const dense2 = tf.layers
      .dense({ units: 10, useBias: true, activation: "tanh" })
      .apply(dense1);

    const adv1 = tf.layers
      .dense({ units: 10, useBias: true, activation: "tanh" })
      .apply(dense2);
    const adv2 = tf.layers
      .dense({ units: 4, useBias: true, activation: "linear" })
      .apply(adv1);

    const val1 = tf.layers
      .dense({ units: 10, useBias: true, activation: "tanh" })
      .apply(dense2);
    const val2 = tf.layers
      .dense({ units: 1, useBias: true, activation: "linear" })
      .apply(val1);

    const output = new DuelingLayer().apply([adv2, val2]);

    model = tf.model({ inputs: input, outputs: output });
  };

  export const fit = async (dataX, dataY) => {
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

    await model.fit(trainX, trainY, {
      batchSize: dataX.length,
      epochs: epochs,
      shuffle: true,
      callbacks: {
        /* onEpochEnd */
      }
    });

    trainX.dispose();
    trainY.dispose();
  };

  export const predict = dataX => {
    let a;
    tf.tidy(() => {
      const X = tf.tensor2d(dataX, [dataX.length, dataX[0].length], "float32");
      a = model.predict(X).arraySync();
    });
    return a;
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
