"""Endpoints for a user to manage their own Canvas API keys."""
from fastapi import APIRouter, Depends, HTTPException, Response
from pydantic import BaseModel, Field

from ..auth import get_current_user
from ..services.canvas_credentials import (
  CanvasCredentialsError,
  delete_credential,
  get_credential_status,
  save_credential,
)

router = APIRouter()


class CanvasCredentialRequest(BaseModel):
  environment: str = Field(pattern="^(development|production)$")
  api_key: str = Field(min_length=1, max_length=10000)


@router.get("")
async def get_canvas_credentials(current_user: dict = Depends(get_current_user)):
  return {"credentials": get_credential_status(current_user["user_id"])}


@router.put("")
async def put_canvas_credential(
  request: CanvasCredentialRequest,
  current_user: dict = Depends(get_current_user),
):
  try:
    save_credential(current_user["user_id"], request.environment, request.api_key)
  except CanvasCredentialsError as exc:
    raise HTTPException(status_code=503, detail=str(exc)) from exc
  return {"credentials": get_credential_status(current_user["user_id"])}


@router.delete("/{environment}", status_code=204)
async def remove_canvas_credential(
  environment: str,
  current_user: dict = Depends(get_current_user),
):
  try:
    delete_credential(current_user["user_id"], environment)
  except CanvasCredentialsError as exc:
    raise HTTPException(status_code=400, detail=str(exc)) from exc
  return Response(status_code=204)
