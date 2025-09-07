From Coq Require Import Reals List.
Import ListNotations.

Module QMC_POVM.
  Parameter V : Type.
  Definition Op := V -> V.

  Parameter Rip : V -> V -> R. (* inner product *)

  Definition snorm (x:V) : R := Rip x x.
  Parameter normalized : V -> Prop.
  Axiom normalized_def : forall psi, normalized psi <-> snorm psi = 1%R.

  Definition prob (E:Op) (psi:V) : R := Rip psi (E psi).

  Fixpoint sumR (xs:list R) : R := match xs with | [] => 0%R | x::xt => x + sumR xt end.

  Parameter sum_effects : list Op -> Op.
  Parameter I : Op. (* identity operator *)

  Axiom prob_sum_effects : forall (Es:list Op) psi,
      prob (sum_effects Es) psi = sumR (map (fun E => prob E psi) Es).

  Axiom prob_identity_norm1 : forall psi, normalized psi -> prob I psi = 1%R.

  Theorem povm_probabilities_sum_to_one : forall (Es:list Op) psi,
      normalized psi ->
      sum_effects Es = I ->
      sumR (map (fun E => prob E psi) Es) = 1%R.
  Proof.
    intros Es psi Hn Heq.
    rewrite <- prob_sum_effects.
    rewrite Heq.
    now apply prob_identity_norm1.
  Qed.

End QMC_POVM.

