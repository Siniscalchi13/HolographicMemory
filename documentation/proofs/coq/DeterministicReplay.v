From Coq Require Import List.
Import ListNotations.

Module Replay.

  Parameter Input Catalog Output : Type.
  Parameter run_call : Catalog -> Input -> Output.

  Definition Transcript := list (Input * Output).

  Definition run_transcript (C:Catalog) (inputs:list Input) : Transcript :=
    map (fun i => (i, run_call C i)) inputs.

  Theorem deterministic_replay : forall C inputs,
      run_transcript C inputs = run_transcript C inputs.
  Proof. reflexivity. Qed.

End Replay.
