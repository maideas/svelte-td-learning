import MazeShell from './MazeShell.svelte';

export const comp = new MazeShell({
	target: document.getElementById("maze-shell-1"),
	props: {
		numX: 12,
		numY: 3,
		numEpisodes: 500,
		planningSteps: 10,  // Dyna-Q parameter

		blocked: Array(),
		terminal: Array(
			[1, 2], [2, 2], [3, 2],
			[4, 2], [5, 2], [6, 2],
			[7, 2], [8, 2], [9, 2],
			[10, 2],
			[11, 2]
		),
		rewards: Array(
			[1, 2, -10.0], [2, 2, -10.0], [3, 2, -10.0],
			[4, 2, -10.0], [5, 2, -10.0], [6, 2, -10.0],
			[7, 2, -10.0], [8, 2, -10.0], [9, 2, -10.0],
			[10, 2, -10.0],
			[11, 2, 0.0]
		),
		defaultReward: -0.1,
		startState: [0, 2]
	}
});

export const comp2 = new MazeShell({
	target: document.getElementById("maze-shell-2"),
	props: {
		numX: 5,
		numY: 5,
		blocked: Array(),
		terminal: Array([1, 1]),
		rewards: Array([1, 1, 1.0]),
		defaultReward: -0.1
	}
});

export const comp3 = new MazeShell({
	target: document.getElementById("maze-shell-3"),
	props: {
		numX: 6,
		numY: 4,
		blocked: Array([0, 0], [3, 2], [4, 1], [3, 3]),
		terminal: Array([1, 1]),
		rewards: Array([1, 1, 1.0]),
		defaultReward: -0.1
	}
});
