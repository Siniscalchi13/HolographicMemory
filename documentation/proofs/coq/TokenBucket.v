From Coq Require Import Arith Lia.

Module TokenBucket.

  Record TB := {
    cap : nat;
    rate : nat (* tokens per tick *)
  }.

  Record State := {
    tokens : nat
  }.

  Definition fill (tb:TB) (s:State) : nat :=
    min (tokens s + rate tb) (cap tb).

  Definition emit (available:nat) : nat := if Nat.leb 1 available then 1 else 0.

  Definition next_state (tb:TB) (s:State) : State :=
    let added := fill tb s in
    let e := emit added in
    {| tokens := added - e |}.

  Fixpoint run (tb:TB) (n:nat) (s:State) : nat :=
    match n with
    | 0 => 0
    | S k =>
        let added := fill tb s in
        let e := emit added in
        e + run tb k {| tokens := added - e |}
    end.

  Lemma emit_le_added : forall a, emit a <= a.
  Proof.
    intro a. unfold emit.
    destruct (Nat.leb 1 a) eqn:E.
    - apply Nat.leb_le in E. lia.
    - apply Nat.leb_gt in E. lia.
  Qed.

  Lemma added_le_tokens_plus_rate : forall tb s,
      fill tb s <= tokens s + rate tb.
  Proof.
    intros. unfold fill. apply Nat.min_lub_l. lia.
  Qed.

  Lemma run_bound_tokens : forall tb n s,
      run tb n s <= tokens s + rate tb * n.
  Proof.
    induction n as [|k IH]; intros s; simpl.
    - lia.
    - set (added := fill tb s).
      set (e := emit added).
      replace (tokens {| tokens := added - e |}) with (added - e) by reflexivity.
      specialize (IH {| tokens := added - e |}).
      assert (Hadded : added <= tokens s + rate tb) by (unfold added; apply added_le_tokens_plus_rate).
      assert (He_le_added : e <= added) by (unfold e; apply emit_le_added).
      assert (He0 : e = 0 \/ e = 1).
      { unfold e, emit. destruct (Nat.leb 1 added); auto. }
      (* Core inequality: e + (added - e) = added *)
      replace (e + run tb k {| tokens := added - e |}) with ((e + (added - e)) + (run tb k {| tokens := added - e |} - (added - e)) + (added - e)).
      2:{ lia. }
      (* Simpler route *)
      clear He0 He_le_added.
      replace (e + run tb k {| tokens := added - e |}) with (added + (run tb k {| tokens := added - e |} - (added - e))).
      2:{ lia. }
      (* Use IH: run tb k s' <= tokens s' + rate*k = (added - e) + rate*k *)
      have Hrun : run tb k {| tokens := added - e |} <= (added - e) + rate tb * k by apply IH.
      assert (H1: run tb (S k) s <= added + rate tb * k) by lia.
      lia using Hadded, H1.
  Qed.

  Corollary run_bound_cap : forall tb n s,
      tokens s <= cap tb ->
      run tb n s <= cap tb + rate tb * n.
  Proof.
    intros. eapply le_trans.
    - apply run_bound_tokens.
    - lia.
  Qed.

End TokenBucket.
