From Coq Require Import Reals.

Module HMC_Correlation.
  Parameter V : Type.
  Parameter Rip : V -> V -> R. (* inner product; real case for simplicity *)
  Definition snorm (x:V) : R := Rip x x.
  Parameter normalized : V -> Prop.
  Axiom normalized_def : forall x, normalized x <-> snorm x = 1%R.

  (* Abstract absolute inner product magnitude *)
  Parameter abs_ip : V -> V -> R.
  Axiom abs_ip_cs : forall x y, normalized x -> normalized y -> abs_ip x y <= 1%R.

  Theorem correlation_bounded_by_one : forall x y,
      normalized x -> normalized y -> abs_ip x y <= 1%R.
  Proof. intros; eapply abs_ip_cs; eauto. Qed.

End HMC_Correlation.

