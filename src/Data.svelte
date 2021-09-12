<script>
  let data = [];
  export let maxData = 1000;

  export const add = item => {
    while (data.length >= maxData) {
      data.shift();
    }
    data.push(item);
  };

  export const getBatch = n => {
    let len = data.length;
    if (len <= n) {
      return [...data];
    }
    let result = new Array(n);
    let taken = new Array(len);
    while (n--) {
      let k = Math.floor(Math.random() * len);
      result[n] = data[k in taken ? taken[k] : k];
      taken[k] = --len in taken ? taken[len] : len;
    }
    return result;
  };

  export const getAll = () => {
    return [...data];
  };

  export const clear = () => {
    data = [];
  };

  export const length = () => {
    return data.length;
  }
</script>
