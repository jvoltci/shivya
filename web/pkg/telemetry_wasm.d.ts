/* tslint:disable */
/* eslint-disable */

export class ShivyaSimulation {
    free(): void;
    [Symbol.dispose](): void;
    add_edge(u_label: string, v_label: string, initial_state: number): void;
    add_vertex(label: string, initial_state: number): number;
    agent_free_energy(obs_0: number, obs_1: number): number;
    agent_update_beliefs(obs_0: number, obs_1: number): Float64Array;
    get_agent_beliefs(): Float64Array;
    get_edge_state(idx: number): number;
    get_edge_u(idx: number): number;
    get_edge_v(idx: number): number;
    get_edges_count(): number;
    get_triangles_count(): number;
    get_vertex_label(idx: number): string;
    get_vertex_state(idx: number): number;
    get_vertices_count(): number;
    constructor();
    reconcile_flows(delta_s: Float64Array): Float64Array;
}

export class SubstrateOrchestrator {
    free(): void;
    [Symbol.dispose](): void;
    constructor();
    step(inputs: Float64Array): string;
}

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
    readonly memory: WebAssembly.Memory;
    readonly __wbg_shivyasimulation_free: (a: number, b: number) => void;
    readonly __wbg_substrateorchestrator_free: (a: number, b: number) => void;
    readonly shivyasimulation_add_edge: (a: number, b: number, c: number, d: number, e: number, f: number) => void;
    readonly shivyasimulation_add_vertex: (a: number, b: number, c: number, d: number) => number;
    readonly shivyasimulation_agent_free_energy: (a: number, b: number, c: number) => number;
    readonly shivyasimulation_agent_update_beliefs: (a: number, b: number, c: number) => [number, number];
    readonly shivyasimulation_get_agent_beliefs: (a: number) => [number, number];
    readonly shivyasimulation_get_edge_state: (a: number, b: number) => number;
    readonly shivyasimulation_get_edge_u: (a: number, b: number) => number;
    readonly shivyasimulation_get_edge_v: (a: number, b: number) => number;
    readonly shivyasimulation_get_edges_count: (a: number) => number;
    readonly shivyasimulation_get_triangles_count: (a: number) => number;
    readonly shivyasimulation_get_vertex_label: (a: number, b: number) => [number, number];
    readonly shivyasimulation_get_vertex_state: (a: number, b: number) => number;
    readonly shivyasimulation_get_vertices_count: (a: number) => number;
    readonly shivyasimulation_new: () => number;
    readonly shivyasimulation_reconcile_flows: (a: number, b: number, c: number) => [number, number];
    readonly substrateorchestrator_new: () => number;
    readonly substrateorchestrator_step: (a: number, b: number, c: number) => [number, number];
    readonly __wbindgen_externrefs: WebAssembly.Table;
    readonly __wbindgen_malloc: (a: number, b: number) => number;
    readonly __wbindgen_realloc: (a: number, b: number, c: number, d: number) => number;
    readonly __wbindgen_free: (a: number, b: number, c: number) => void;
    readonly __wbindgen_start: () => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;

/**
 * Instantiates the given `module`, which can either be bytes or
 * a precompiled `WebAssembly.Module`.
 *
 * @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
 *
 * @returns {InitOutput}
 */
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
 * If `module_or_path` is {RequestInfo} or {URL}, makes a request and
 * for everything else, calls `WebAssembly.instantiate` directly.
 *
 * @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
 *
 * @returns {Promise<InitOutput>}
 */
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
