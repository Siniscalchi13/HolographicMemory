From Coq Require Import Arith.

Module ACLTermination.

  Variable M : nat -> Prop.
  (* We model measure as a natural number; each accepted transformation strictly decreases it. *)

  Inductive step : nat -> nat -> Prop :=
  | Decrease : forall n, n > 0 -> step n (n-1).

  Theorem terminates : forall n, exists m, step^* n m /\ (forall k, step m k -> False).
  Proof.
    induction n using lt_wf_ind.
    intros.
    destruct n.
    - exists 0. split.
      + constructor. (* reflexive transitive closure base *)
      + intros k Hk. inversion Hk; lia.
    - (* n = S p *)
      assert (Hdec: step (S n0) n0) by (constructor; lia).
      specialize (H n0 (lt_n_Sn n0)).
      destruct H as [m [Hstar Hterm]].
      exists m. split.
      + eapply rt_trans. 2: exact Hstar. apply rt_step. exact Hdec.
      + exact Hterm.
  Qed.

End ACLTermination.
