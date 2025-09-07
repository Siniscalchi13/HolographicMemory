From Coq Require Import Reals Lra.
Local Open Scope R_scope.

Module MSC_Selection.
  (* Score function components *)
  Record Params := {
    lam : R; rho : R; gamma : R; eta : R
  }.

  (* Overflow penalty modeled as max(0, tau - Lm). *)
  Definition cap_pen (tau Lm:R) : R := Rmax 0 (tau - Lm).

  (* Abstract score model: J = comp - lam*c + rho*p + gamma*align - eta*cap_pen + bonus *)
  Definition J (comp cost plat align tau Lm bonus:R) (p:Params) : R :=
    comp - (lam p) * cost + (rho p) * plat + (gamma p) * align - (eta p) * (cap_pen tau Lm) + bonus.

  (* MSC-1: Monotonicity in competence when other terms fixed *)
  Lemma J_monotone_in_comp : forall p c pl a tau Lm b x y,
      x <= y -> J x c pl a tau Lm b p <= J y c pl a tau Lm b p.
  Proof. intros; unfold J; lra. Qed.

  (* MSC-2: For tau1 <= tau2 with both above Lm, score decreases by at least eta*(tau2 - tau1). *)
  Lemma cap_pen_linear_above : forall tau Lm, tau >= Lm -> cap_pen tau Lm = tau - Lm.
  Proof.
    intros. unfold cap_pen. rewrite Rmax_right; lra.
  Qed.

  Lemma J_decreases_with_tau_above_Lm : forall p comp cost plat align Lm bonus tau1 tau2,
      tau1 >= Lm -> tau2 >= Lm -> tau1 <= tau2 ->
      J comp cost plat align tau2 Lm bonus p = J comp cost plat align tau1 Lm bonus p - (eta p) * (tau2 - tau1).
  Proof.
    intros. unfold J.
    repeat rewrite cap_pen_linear_above by assumption.
    ring.
  Qed.

End MSC_Selection.

