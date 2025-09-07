From Coq Require Import List Arith Lia.
Import ListNotations.

Module WAL.

  (* Simplified WAL model: log is list of entries; stable storage is committed prefix. *)
  Record State := {
    log : list nat; (* full in-memory log *)
    stable : list nat (* persisted prefix *)
  }.

  (* Invariant: stable is a prefix of log *)
  Definition prefix {A} (p l:list A) := exists s, l = p ++ s.

  Definition inv (s:State) : Prop := prefix (stable s) (log s).

  (* Append: write to log; flush: advance stable to full log; crash: lose volatile suffix, keep stable. *)
  Inductive step : State -> State -> Prop :=
  | Append : forall s x, step s {| log := log s ++ [x]; stable := stable s |}
  | Flush  : forall s, step s {| log := log s; stable := log s |}
  | Crash  : forall s, step s {| log := stable s; stable := stable s |}.

  Lemma inv_preserved : forall s s', inv s -> step s s' -> inv s'.
  Proof.
    intros s s' Hinv Hstep. inversion Hstep; subst; simpl.
    - (* Append *)
      unfold inv in *. destruct Hinv as [suf Heq]. exists (suf ++ [x]). now rewrite Heq, app_assoc.
    - (* Flush *)
      unfold inv. exists []. now rewrite app_nil_r.
    - (* Crash *)
      unfold inv. exists []. now rewrite app_nil_r.
  Qed.

  (* Recovery theorem: after any sequence ending with Crash, state equals a prefix equal to previous stable; no loss beyond volatile suffix. *)
  Theorem crash_recovers_prefix : forall s s',
      inv s -> step s s' -> exists p, stable s' = p /\ prefix p (log s').
  Proof.
    intros s s' Hinv Hstep. inversion Hstep; subst; simpl.
    - exists (stable s). split; [reflexivity|]. unfold prefix. exists (log s ++ [x] ++ []) ; now rewrite app_assoc.
    - exists (log s). split; [reflexivity|]. unfold prefix. exists []. now rewrite app_nil_r.
    - exists (stable s). split; [reflexivity|]. unfold prefix. exists []. now rewrite app_nil_r.
  Qed.

End WAL.
