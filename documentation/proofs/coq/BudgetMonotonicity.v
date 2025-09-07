From Coq Require Import Arith Lia.

Module Budget.

  Record Budgets := { budget : nat }.

  Inductive step_reservation : Budgets -> Budgets -> Prop :=
  | Reserve : forall b,
      budget b > 0 ->
      step_reservation b {| budget := budget b - 1 |}.

  Inductive step_commit : Budgets -> Budgets -> Prop :=
  | CommitSuccess : forall b,
      budget b > 0 ->
      step_commit b {| budget := budget b - 1 |}
  | CommitFail : forall b,
      step_commit b b.

  Definition non_increasing (f : Budgets -> Budgets -> Prop) : Prop :=
    forall b b', f b b' -> budget b' <= budget b.

  Theorem reservation_non_increasing : non_increasing step_reservation.
  Proof. intros b b' H. inversion H; subst; simpl; lia. Qed.

  Theorem commit_non_increasing : non_increasing step_commit.
  Proof.
    intros b b' H; inversion H; subst; simpl; lia.
  Qed.

End Budget.
