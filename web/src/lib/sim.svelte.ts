export const sim = $state({
  playing: false,
  speed: 1,
  time: 0
});

export function pauseSim(): void {
  sim.playing = false;
}

export function toggleSim(): void {
  sim.playing = !sim.playing;
}

export function setSpeed(speed: number): void {
  sim.speed = speed;
}
